import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch._dynamo.config
from torch.nn.attention import SDPBackend, sdpa_kernel
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm import tqdm

from .dataset import load_imagenet
from .log.loggers import MetricLogger
from .models.blocks.ema import EMA, EMAWrapper
from .models.ssdd.losses import GanLoss, SSDDLosses
from .models.ssdd.ssdd import DC_SSDAE
from .mutils.main_utils import (
    TaskState,
    UpAccelerator,
    ensure_path,
    split_dict,
)
from .mutils.torch_utils import count_parameters, freeze_model, reproducible_rand, unwrap
from .mutils.train_utils import (
    aggregate_losses,
    auto_compile,
    build_optimizer,
    load_training_state,
    save_training_state,
)

####################################################################
# Common initialization
####################################################################


class AutoencodingTasks:
    SHOW_MODEL_PARTS = ["encoder", "decoder"]

    def __init__(self, cfg):
        self.state = TaskState(cfg=cfg)
        self.optimizer = None
        self.gan_optimizer = None
        self.training = False

        self.setup_job_env()
        self.setup()

        self.load_data()
        self.load_models()
        self.show_model()

        if self.training:
            self.task_train_prepare()
            # Load training state once model & optimizer are ready
            load_training_state(self.state, self.cfg.checkpoint_path)

        self.accelerator.wait_for_everyone()

    @property
    def cfg(self):
        return self.state.cfg

    @property
    def accelerator(self):
        return self.state.accelerator

    @property
    def models(self):
        return self.state.models

    @property
    def logger(self):
        return self.state.logger

    def print(self, *args, **kwargs):
        self.state.accelerator.print(*args, **kwargs)

    def run(self, task_name=None):
        task_name = task_name or self.cfg.task
        method_name = f"task_{task_name}"
        if hasattr(self, method_name):
            return self.__getattribute__(method_name)()
        else:
            raise ValueError(f"Run function {method_name} for task {task_name} inside {self.__class__.__name__} not found")

    def __call__(self, task_name=None):
        # Run task
        with self.logger.on_task_run() as task_log:
            set_seed(self.cfg.seed)
            task_result = self.run(task_name)

            # End task
            task_log.results = task_result
        self.accelerator.end_training()

        return task_result

    ##### Setup #####

    def setup_job_env(self):
        """Setup the job environment"""
        # For easier debugging
        #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
        #torch.autograd.set_detect_anomaly(True)	# Debug NaN / Inf in backward pass
        
        # Ensure working inside the run directory
        ensure_path(self.cfg.run_dir)
        os.chdir(self.cfg.run_dir)
        ensure_path(self.cfg.cache_dir)
        ensure_path(self.cfg.checkpoint_path)

        # Set seed
        set_seed(self.cfg.seed)

        # Set torch configuration
        torch.set_default_dtype(torch.float32)
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.optimize_ddp = False
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def setup(self):
        """Setup the job modules"""

        ### Prepare accelerator and DDP ###
        self.state.accelerator = UpAccelerator(
            kwargs_handlers=[DistributedDataParallelKwargs()],
            gradient_accumulation_steps=self.cfg.training.grad_accumulate,
            step_scheduler_with_optimizer=False,
            mixed_precision=(self.cfg.training.mixed_precision or "no"),
        )

        # Display job information
        self.init_state()

        # Logger
        self.state.logger = self.build_task_logger()

        # Accelerate configuration
        get_logger("accelerate.accelerator").setLevel("WARNING")
        get_logger("accelerate.checkpointing").setLevel("WARNING")

    def init_state(self):
        """Initialize the configuration & state with job-specific settings"""
        cfg = self.cfg

        # Initialize state variables
        self.state.num_processes = self.accelerator.num_processes
        self.accelerator.register_for_checkpointing(self.state)
        self.state.cur_epoch = 0
        self.state.cur_steps = 0

        self.opti_models = []  # pylint: disable=W0201
        self.training = cfg.task.endswith("train")  # pylint: disable=W0201
        self.checkpoint_path = Path(cfg.checkpoint_path)  # pylint: disable=W0201

    ##### Build modules #####

    def load_data(self):
        (train_dataset, test_dataset), (self.train_loader, self.test_loader) = load_imagenet(self.cfg.dataset)
        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare_test_data(self.test_loader)

        self.print("Loaded ImageNet dataset:", {"train": train_dataset, "test": test_dataset})

    def prepare_model(
        self,
        model: torch,
        *,
        name: str,
        compile: bool = False,
        ema: Optional[Mapping] = None,
        freeze: Optional[bool] = None,
        remove_from_checkpointing: bool = False,
    ):
        """Utility to load a model, freeze it if not main model, etc."""
        if freeze is None:
            freeze = not self.training

        # 1. Wrap model
        use_ema = False
        if ema is not None and ema.decay and self.training:
            use_ema = True
            model = EMAWrapper(model, **ema)

        # 2. Set model & gradients state
        model.train(self.training)
        if freeze:
            model.requires_grad_(False)

        # 3. Wrap with accelerator
        prep_modules = [model] if not use_ema else [model.model, model.ema.ema_model]
        for i, m in enumerate(prep_modules):
            m = self.accelerator.prepare(m)
            if remove_from_checkpointing:
                self.accelerator._models.remove(m)
            prep_modules[i] = m

        if not remove_from_checkpointing:
            self.state.registered_models.append(name)
        if use_ema:
            model.model = prep_modules[0]
            model.ema.ema_model = prep_modules[1]
            if not remove_from_checkpointing:
                self.state.registered_models.append(name + "_ema")
        else:
            model = prep_modules[0]

        # 4. Compile if needed
        model = auto_compile(compile, model)

        self.models[name] = model
        return model

    def build_model(self, maker, **kwargs):
        prep_args = ["name", "compile", "ema", "freeze", "model_init", "remove_from_checkpointing"]
        prep_kwargs, module_kwargs = split_dict(kwargs, prep_args)
        model = maker(**module_kwargs)
        return self.prepare_model(model, **prep_kwargs)

    def load_models(self):
        self.build_model(DC_SSDAE, name="ae", **self.cfg.dc_ssdae)
        if self.training:
            self.build_model(SSDDLosses, name="aux_losses", **self.cfg.aux_losses, ae=self.models["ae"], accelerator=self.accelerator, checkpoint=self.cfg.dc_ssdae.checkpoint)

            if self.cfg.get("gan", None) is not None:
                model_gan = GanLoss(**self.cfg.gan)
                self.prepare_model(model_gan, name="gan")

            if self.cfg.distill_teacher:
                self.build_model(DC_SSDAE, name="teacher", **self.cfg.dc_ssdae, remove_from_checkpointing=True)
                freeze_model(self.models["teacher"])
                self.models["teacher"].train()

    def task_train_prepare(self):
        self.optimizer = build_optimizer([self.models["ae"], self.models["aux_losses"]], self.cfg.training.lr, self.cfg.training.weight_decay)
        self.print(f"Optimizer for autoencoder: {self.optimizer}")
        self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.cfg.get("gan", None) is not None:
            self.gan_optimizer = build_optimizer([self.models["gan"]], self.cfg.training.lr, self.cfg.training.weight_decay)
            self.print(f"Optimizer for GAN: {self.gan_optimizer}")
            self.gan_optimizer = self.accelerator.prepare(self.gan_optimizer)

    ##### Logging #####

    def build_task_logger(self):
        return MetricLogger(self.state, train=self.training)

    def _show_model_subparameter_count(self, model, recursive_on=None, depth=-1, name=None):
        if depth == -1:
            model = unwrap(model, unw_ema=True)
            name = name or "model"
            self.print(f"{name} parameters count:")
            self.print(f"Total: #{count_parameters(model)}   (trainable: #{count_parameters(model, trainable=True)})")
            self._show_model_subparameter_count(model, recursive_on, depth=0, name=name)
        else:
            for name, m in unwrap(model).named_children():
                self.print("     " * depth + f"- {name}: #{count_parameters(m)}   (trainable: #{count_parameters(m, trainable=True)})")
                if recursive_on and name in recursive_on:
                    sub_rec = [n[len(name) + 1 :] if n.startswith(name + ".") else n for n in recursive_on]
                    self._show_model_subparameter_count(m, sub_rec, depth=depth + 1, name=name)

    def show_model(self):
        for m_name, m in self.models.items():
            if self.SHOW_MODEL_PARTS not in [None, False]:
                self._show_model_subparameter_count(m, recursive_on=self.SHOW_MODEL_PARTS, name=m_name)
            self.print(f"{m_name}:", m)

    ##### Training (task_train) #####

    def set_train_state(self, is_training=True):
        for m in self.models.values():
            m.train(is_training)

        optimizers = [self.optimizer, self.gan_optimizer]
        for opt in optimizers:
            if opt is not None and hasattr(opt, "train"):
                if is_training:
                    opt.train()
                else:
                    opt.eval()

    def _compute_train_loss(self, batch, train_ctx):
        x, _ = batch
        target_x = None

        # Train SSDD: main loss & predict x0
        if "teacher" in self.models:
            # Train on distillation
            with torch.no_grad():
                teacher_gen = self.models["teacher"](x)
            target_x = teacher_gen.x0_pred
            ssdd_out = self.models["ae"](target_x, z=teacher_gen.z, noise=teacher_gen.noise)
        else:
            # Train on target
            ssdd_out = self.models["ae"](x)
        losses = ssdd_out.losses

        # Add auxiliary losses
        aux_losses = self.models["aux_losses"](x, ssdd_out.x0_pred, target_x=target_x)
        losses.update(aux_losses)

        # Add GAN losses
        if "gan" in self.models:
            losses.update(
                self.models["gan"](
                    x_gt=x if target_x is None else target_x,
                    x_pred=ssdd_out.x0_pred,
                    xt=ssdd_out.xt,
                    t=ssdd_out.t,
                    existing_losses=losses,
                    n_train_steps=train_ctx["cur_steps"],
                    step="disc_loss",
                )
            )
            train_ctx["gan_ctx"] = {
                "x_pred": ssdd_out.x0_pred,
                "xt": ssdd_out.xt,
                "t": ssdd_out.t,
            }

        return losses

    def _compute_train_gan_loss(self, batch, train_ctx):
        x, _ = batch
        return self.models["gan"](
            x_gt=x,
            **train_ctx["gan_ctx"],
            n_train_steps=train_ctx["cur_steps"],
            step="train",
        )

    def _train_do_step(self, optimizer: torch.optim.Optimizer, batch: Any, train_ctx: Dict[str, Any], step_gan=False):
        acc = self.accelerator

        if step_gan:
            models = [self.models["gan"]]
            with acc.autocast():  # pylint: disable=no-member
                losses = self._compute_train_gan_loss(batch, train_ctx)
        else:
            models = [self.models["ae"], self.models["aux_losses"]]
            with acc.autocast():  # pylint: disable=no-member
                losses = self._compute_train_loss(batch, train_ctx)

        assert isinstance(losses, dict) and all(isinstance(v, torch.Tensor) for v in losses.values()), f"Losses should be a dict of tensors, got {losses}"
        assert len(losses) > 0, "No loss returned"

        sum_loss, losses = aggregate_losses(self.cfg, losses)

        acc.backward(sum_loss)

        if self.cfg.training.grad_clip and acc.sync_gradients:
            for m in models:
                acc.clip_grad_norm_(m.parameters(), self.cfg.training.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # Keep here to ensure grad accumulation works correctly

        return {k: v.detach() for k, v in losses.items()}

    def _task_train_one_epoch(self):
        """Generic training loop. Can be overwritten for more specific tasks. If used, you need to define _compute_train_loss"""
        acc = self.accelerator
        assert self.training, "Not in training mode"
        self.set_train_state()

        with self.logger.on_epoch(self.train_loader):
            for i_batch, batch in enumerate(self.train_loader):
                # Start step
                with self.logger.on_batch(i_batch) as batch_log:
                    train_ctx = {
                        "losses": batch_log.losses,
                        "i_batch": i_batch,
                        "cur_epoch": self.state.cur_epoch,
                        "cur_steps": self.state.cur_steps,
                    }

                    with acc.accumulate(self.models["ae"], self.models["aux_losses"]):
                        batch_log.losses.update(self._train_do_step(self.optimizer, batch, train_ctx=train_ctx))

                    # EMA
                    if EMA.uses_ema(self.models["ae"]):
                        self.set_train_state(False)
                        EMA.update_ema_modules(self.models["ae"])
                        self.set_train_state(True)

                    # GAN step
                    if "gan" in self.models:
                        with acc.accumulate(self.models["gan"]):
                            batch_log.losses.update(self._train_do_step(self.gan_optimizer, batch, train_ctx=train_ctx, step_gan=True))

                acc.wait_for_everyone()
                self.state.cur_steps += 1

    def _task_train_post_eval(self, did_eval):
        # Storing last checkpoint
        ckpt = self.checkpoint_path / "last"
        self.accelerator.print(f"Storing model checkpoint inside {ckpt}")
        save_training_state(self.state, ckpt)

        if did_eval and self.logger.epochs_since_best_score() == 0:
            ckpt = self.checkpoint_path / "best"
            self.accelerator.print(f"Best {self.cfg.training.save_on_best} so far, storing a copy of the model checkpoint to {ckpt}")
            save_training_state(self.state, ckpt)

    def task_train(self):
        # Start directly at state.cur_epoch (even if > 0)
        did_eval_last_epoch = False
        # To avoid SDPBackend.EFFICIENT_ATTENTION RuntimeError: Function 'ScaledDotProductEfficientAttentionBackward0' returned nan values in its 0th output.
        # Use SDPBackend.CUDNN_ATTENTION for NVIDIA Hopper architectures (e.g., H100 GPUs), SDPBackend.FLASH_ATTENTION for NVIDIA Ampere architectures (e.g., A100 GPUs)
        backend = SDPBackend(self.cfg.training.sdpa_kernel) if isinstance(self.cfg.training.sdpa_kernel, int) else SDPBackend.MATH
        with sdpa_kernel(backend):
            while self.state.cur_epoch < self.cfg.training.epochs:
                self._task_train_one_epoch()
                
                # Evaluation
                eval_now = (self.state.cur_epoch + 1) % self.cfg.training.eval_freq == 0
                if eval_now:
                    did_eval_last_epoch = True
                    self.task_eval()

                self.state.cur_epoch += 1
                self._task_train_post_eval(did_eval_last_epoch)

        # Ensure last eval
        if not did_eval_last_epoch:
            self.accelerator.print("Training stopped, final evaluation")
            self.task_eval()
            self._task_train_post_eval(did_eval_last_epoch)

    ##### Evaluation (task_eval) #####

    def _generate_for_eval(self, x, y=None, generator=None):
        noise = reproducible_rand(self.accelerator, generator, x.shape)
        steps = 1 if "teacher" in self.models else None
        gen_x = self.models["ae"](x, noise=noise, steps=steps)
        return gen_x

    @torch.no_grad()
    def task_eval(self):
        acc = self.accelerator
        self.set_train_state(False)
        self.generator = torch.Generator(device=acc.device)
        self.generator.manual_seed(self.cfg.seed)

        with self.logger.on_eval(self.test_loader) as eval_log:
            # Eval reconstruction of test set
            enum_tests = tqdm(self.test_loader, desc="Reconstructing from test set", disable=not acc.is_main_process)
            for test_samples, label in enum_tests:
                test_samples = test_samples.to(acc.device)
                with acc.autocast():
                    rec_samples = self._generate_for_eval(test_samples, generator=self.generator)

                rgb_test_samples = self.to_rgb(test_samples)
                rgb_rec_samples = self.to_rgb(rec_samples)

                self.logger.metrics.update(
                    x_gt=rgb_test_samples,
                    x_pred=rgb_rec_samples,
                    y_gt=label,
                )

            # Generate displayed samples
            if self.cfg.show_samples and acc.is_main_process:
                n_samples = self.cfg.show_samples
                eval_log.gt_samples = rgb_test_samples[:n_samples]
                eval_log.rec_samples = rgb_rec_samples[:n_samples]

        return deepcopy(self.logger.metrics.last_m_vals)

    ##### Other tasks #####

    @torch.no_grad()
    def task_z_stats(self):
        acc = self.accelerator
        self.models["ae"].eval()
        assert acc.is_main_process, "Z stats can only be computed on the main process"

        z_dataset = []

        enum_tests = tqdm(self.test_loader, desc="Reconstructing from test set", disable=not acc.is_main_process)
        for x, _ in enum_tests:
            x = x.to(acc.device)
            with acc.autocast():
                z = self.models["ae"].encode(x).mode()
            z = z.to(torch.float64)

            z_dataset.append(z.cpu())

        z_dataset = torch.cat(z_dataset, dim=0)
        z_mean = z_dataset.mean()
        z_std = z_dataset.std(unbiased=True)

        z_mean, z_std = z_mean.item(), z_std.item()
        acc.print(f"Z stats: z_mean={z_mean:.5f} z_std={z_std:.5f}")
        return {"z_mean": z_mean, "z_std": z_std}

    ##### Utils #####

    def to_rgb(self, x):  # x in [-1;1] ; output will be in [0;1]
        assert x.ndim == 4
        return torch.clamp(255 * (x + 1) / 2, 0, 255).round() / 255
