import json
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from ..mutils.main_utils import ensure_path
from ..mutils.timers import TimersManager
from ..mutils.train_utils import aggregate_losses
from .metrics import MetricsManager
from .visualize import (
    show_generation_result,
)

####################################################################
# Logging utilities
####################################################################


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4g} ({global_avg:.4g})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class SmoothedMetrics:
    def __init__(self):
        self.metrics = {}

    def create(self, metric_name, **kwargs):
        self.metrics[metric_name] = SmoothedValue(**kwargs)

    def update(self, n=1, **metrics):
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = SmoothedValue(fmt="{global_avg:g}")
            self.metrics[name].update(value, n=n)
        return {name: self.get(name) for name in self.metrics.keys()}

    def get(self, name):
        return self.metrics[name]


def convert_json(obj):
    if isinstance(obj, (list, tuple)):
        return [convert_json(item) for item in obj]
    elif isinstance(obj, Mapping):
        return {k: convert_json(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.cpu().tolist()
    elif hasattr(obj, "to_json"):
        return convert_json(obj.to_json())
    else:
        return obj


####################################################################
# Logging results
####################################################################


@dataclass
class TaskResult:
    results = None


@dataclass
class EvalResult:
    rec_samples: torch.Tensor = None
    gt_samples: torch.Tensor = None


@dataclass
class BatchResult:
    losses: dict = field(default_factory=dict)


####################################################################
# Logging manager
####################################################################


class MetricLogger:
    def __init__(self, state, train):
        self.state = state
        self.eps = 1e-6
        self.timers = TimersManager("train", "epoch", "eval", "total")
        self.train = train
        self.metrics = MetricsManager(self.cfg, self.accelerator.device)
        self.smooth = SmoothedMetrics()
        self.score_history = []
        self.epoch_data = {}

        self.timers.total.start()  # pylint: disable=no-member

        if self.accelerator.is_main_process and train:
            self.writer = SummaryWriter(log_dir="tensorboard_logs")
            self.print(f"Will write tensorboard logs inside {Path(self.writer.log_dir).resolve()}")
        else:
            self.writer = None

        self.log_job_details()

    @property
    def cfg(self):
        return self.state.cfg

    @property
    def accelerator(self):
        return self.state.accelerator

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def log_job_details(self):
        """Log job informations before building"""
        cfg, accelerator = self.cfg, self.accelerator
        accelerator.wait_for_everyone()
        self.print(f"Runtime at {cfg.runtime_path}")
        self.print(f"Running inside {cfg.run_dir}")
        self.print(f"Running args: {sys.argv}")
        self.print("Command:", " ".join([f"'{arg}'" for arg in sys.argv]))
        self.print(f"Accelerator with {accelerator.num_processes} processes, running on {accelerator.device}")
        self.print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}\n\n")
        with open("config.yaml", "w") as f_cfg:
            f_cfg.write(OmegaConf.to_yaml(cfg))

        accelerator.wait_for_everyone()

    def state_dict(self):
        return {
            "timers": self.timers.state_dict(),
            "metrics": self.metrics.state_dict(),
            "score_history": self.score_history,
            "cur_epoch": self.state.cur_epoch,
            "cur_step": self.state.cur_steps,
            "epoch_data": self.epoch_data,
        }

    def load_state_dict(self, state):
        self.timers.load_state_dict(state["timers"])
        self.metrics.load_state_dict(state["metrics"])
        self.score_history = state["score_history"]
        self.state.cur_epoch = state["cur_epoch"]
        self.state.cur_steps = state["cur_step"]
        self.epoch_data = state["epoch_data"]

    def epochs_since_best_score(self):
        if self.score_history:
            return np.argmin(self.score_history[::-1])
        return -1

    # ===== Task logging =====

    @contextmanager
    def on_task_run(self):
        ### Start Task ###
        self.print(f"====================== RUNNING TASK {self.cfg.task}")
        self.accelerator.wait_for_everyone()
        cfg = self.cfg
        acc = self.accelerator
        if self.train:
            acc.print("Starting training")
        else:
            acc.print("Starting evaluation")

        gpu_batch_size = cfg.dataset.batch_size // acc.num_processes
        acc.print(f"Batch size of {cfg.dataset.batch_size} ({gpu_batch_size} per GPU, {cfg.training.grad_accumulate} acumulation step(s) {acc.num_processes} process(es))")

        yield (ret := TaskResult())

        ### End Task ###

        results = ret.results

        if self.train:
            self.print(f"[T_total={self.timers.total} | T_train={self.timers.train}] Training done")  # pylint: disable=no-member
        else:
            self.print(f"[T_total={self.timers.total}] Evaluation done")  # pylint: disable=no-member
        if self.writer is not None:
            self.writer.close()

        self.accelerator.wait_for_everyone()
        self.print("Task done - End of job - Success - Exiting")

        if results is not None and self.accelerator.is_main_process:
            # Add logg informations
            if not isinstance(results, dict):
                results = {"result": results}

            results = {**results}
            if hasattr(self.cfg, "_cur_epoch"):
                results["epoch"] = self.state.cur_epoch
            if hasattr(self.cfg, "_cur_steps"):
                results["steps"] = self.state.cur_steps

            result_path = Path("task_result.json").resolve()
            result_arr = []
            if result_path.exists():
                with result_path.open("r") as f:
                    try:
                        result_arr = json.load(f)
                    except json.JSONDecodeError:
                        self.print(f"Could not load {result_path} - creating a new one")
            result_arr.append(results)

            with result_path.open("w") as f:
                try:
                    json.dump(convert_json(result_arr), f)
                except TypeError as e:
                    self.print("Evaluation result:", result_arr)
                    self.print(f"Could not save evaluation result to JSON: error {e}")
                else:
                    self.print(f"Saved task result to {result_path}")

    # ===== Train epoch logging =====

    @contextmanager
    def on_epoch(self, train_loader):
        with torch.no_grad():
            self.print("---\n\n")
            self.print(
                f"[T_total={self.timers.total} | T_train={self.timers.train}]"  # pylint: disable=no-member
                f" Start epoch {self.state.cur_epoch}"
            )

            self.epoch_data["n_iters"] = 0
            self.epoch_data["sum_loss"] = 0
            self.epoch_data["sum_all_losses"] = {}
            self.epoch_data["n_batches"] = len(train_loader)

        with self.timers.epoch(reset=True), self.timers.train:  # pylint: disable=no-member
            yield

        with torch.no_grad():
            mean_loss = self.epoch_data["sum_loss"] / self.epoch_data["n_iters"]
            mean_all_losses = {k: v / self.epoch_data["n_iters"] for k, v in self.epoch_data["sum_all_losses"].items()}
            losses_str = " ; ".join([f"{k}={v:g}" for k, v in mean_all_losses.items()])
            if self.cfg.training.save_on_best == "loss":
                self.score_history.append(mean_loss)

            self.print(
                f"[T_total={self.timers.total} | T_train={self.timers.train} | T_epoch={self.timers.epoch}]"  # pylint: disable=no-member
                f" End of epoch {self.state.cur_epoch} ({self.state.cur_steps} steps) train loss {mean_loss:g}"
            )
            self.print(f"[Epoch {self.state.cur_epoch}] All losses: [[{losses_str}]]")

            if self.writer is not None:
                self.writer.add_scalar("Loss/average", mean_loss, self.state.cur_steps)
                self.writer.add_scalar("hparam/step_elapsed_epochs", self.state.cur_epoch + 1, self.state.cur_steps)
                for k, v in mean_all_losses.items():
                    self.writer.add_scalar(f"Loss/{k}", v, self.state.cur_steps)
                self.writer.flush()

    @contextmanager
    def on_batch(self, i_batch):
        yield (ret := BatchResult())
        losses = ret.losses

        with torch.no_grad():
            # Aggregate losses
            sum_loss, losses = aggregate_losses(self.cfg, losses)
            sum_loss = self.accelerator.gather(sum_loss).mean()
            losses = {k: self.accelerator.gather(l).mean() for k, l in sorted(losses.items())}

            # Convert losses to float
            if isinstance(sum_loss, torch.Tensor):
                sum_loss = float(sum_loss.item())
            losses = {k: float(l.item()) for k, l in losses.items()}
            smooth_losses = self.smooth.update(**losses)
            smooth_sum_loss = self.smooth.update(sum_loss=sum_loss)["sum_loss"]

            self.epoch_data["sum_loss"] += sum_loss
            self.epoch_data["n_iters"] += 1
            for k, v in losses.items():
                self.epoch_data["sum_all_losses"][k] = self.epoch_data["sum_all_losses"].get(k, 0) + v

            if self.writer is not None:
                self.writer.add_scalar("batch_loss/average", sum_loss, self.state.cur_steps)
                for k, v in losses.items():
                    self.writer.add_scalar(f"batch_loss/{k}", v, self.state.cur_steps)

            if i_batch % self.cfg.training.log_freq == 0:
                losses_str = " ; ".join([f"{k}={v}" for k, v in smooth_losses.items()])
                self.accelerator.print_nolog(
                    "\033[K"
                    f"[T_total={self.timers.total} | T_train={self.timers.train} | T_epoch={self.timers.epoch}] "  # pylint: disable=no-member
                    f"Epoch {self.state.cur_epoch}, batch {i_batch + 1} / {self.epoch_data['n_batches']} "
                    f"(step {self.state.cur_steps}) "
                    f"loss={smooth_sum_loss} (avg={self.epoch_data['sum_loss'] / self.epoch_data['n_iters']:.4g}) "
                    f"[[all losses: {losses_str}]]",
                    end="\n",
                    flush=True,
                )

    # ===== eval logging =====

    @contextmanager
    def on_eval(self, data_loader):
        self.n_total_batches = len(data_loader)
        self.metrics.reset()

        with self.timers.eval:  # pylint: disable=no-member
            yield (ret := EvalResult())

        with self.timers.eval:  # pylint: disable=no-member
            self.accelerator.wait_for_everyone()
            last_m_values = self.metrics.compute(store_best=True)  # Ensure we compute the metrics for the step, and store best ones

            self.print(f"[Epoch {self.state.cur_epoch}] Test metrics:", self.metrics.metrics_as_str())
            if self.writer is not None:
                for name, v in last_m_values.items():
                    self.writer.add_scalar(f"metric/{name}", v, self.state.cur_steps)
            if self.train:
                self.print(f"[Epoch {self.state.cur_epoch}] Best metrics:", self.metrics.bests_as_str())

                if self.cfg.training.save_on_best != "loss":
                    self.score_history.append(last_m_values[self.cfg.training.save_on_best])

            ### Show images ###
            self.show_images = self.accelerator.is_main_process and ret.rec_samples is not None

            if self.show_images:
                self.accelerator.debug("Writing images to disk...")
                plot_path = Path("plots").resolve()
                ensure_path(plot_path)

                fig_suffix = f"epoch={self.state.cur_epoch}" if self.train else ""
                if hasattr(self.cfg, "_eval_suffix"):
                    fig_suffix += f"_{self.state.eval_suffix}"

                # Show image decoding steps
                title = f"Generated samples after epoch {self.state.cur_epoch}"
                if isinstance(ret.rec_samples, torch.Tensor):
                    fig = show_generation_result(ret.rec_samples, ret.gt_samples, title=title)
                else:
                    raise NotImplementedError("Unknown type of displayed samples")
                fig_name = f"generation{fig_suffix}"
                if self.writer:
                    self.writer.add_figure("HR/step_decode", fig, global_step=self.state.cur_steps)
                else:
                    fig.savefig(f"{plot_path}/{fig_name}.png", dpi=100)
                    self.print(f"Saved generation steps figure at {plot_path}/{fig_name}.png")

                self.accelerator.debug("Image(s) saved on disk")

            if self.writer:
                self.writer.flush()
            self.accelerator.wait_for_everyone()

            self.print(f"End of epoch timers: [{self.timers.join_str(' | ')}]")
