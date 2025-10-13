import torch


class TimeSamplerLogitNormal:
    def __init__(self, t_mean=0, t_std=1.0):
        self.t_std = t_std
        self.t_mean = t_mean

    def __call__(self, batch_size, device):
        t = torch.randn(batch_size, device=device) * self.t_std + self.t_mean
        return torch.sigmoid(t)


class FlowMatchingTrainer:
    def __init__(
        self,
        *,
        timescale: float = 1_000,
        sigma_min: float = 0.0,
        t_sampler_args=None,
    ):
        self.prediction_type = None
        self.t_sampler = TimeSamplerLogitNormal(**(t_sampler_args or {}))

        # Args
        self.timescale = timescale
        self.sigma_min = sigma_min

    def alpha(self, t):
        return 1.0 - t

    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)

    def A(self, t):
        return 1.0

    def B(self, t):
        return -(1.0 - self.sigma_min)

    def add_noise(self, x, t, noise=None):
        # t=0.0 -> no noise ; t=1.0 -> full noise
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        x_t = x_t

        return x_t, noise

    def loss(self, fn, x, fn_kwargs=None, noise=None):
        if fn_kwargs is None:
            fn_kwargs = {}

        t = self.t_sampler(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t, noise=noise)

        v_pred = fn(x_t, t=t * self.timescale, **fn_kwargs)

        target = self.A(t) * x + self.B(t) * noise  # -dxt/dt
        target = target

        loss = ((v_pred.float() - target.float()) ** 2).mean()
        return loss, (x_t, noise, t, v_pred)

    def get_prediction(
        self,
        fn,
        x_t,
        t,
        fn_kwargs=None,
    ):
        return fn(x_t, t=t * self.timescale, **(fn_kwargs or {}))

    def step(self, x_t, v_pred, cur_t, next_t=0):
        if not isinstance(v_pred, torch.Tensor):
            v_pred = torch.tensor(v_pred, device=x_t.device)
        cur_t = cur_t.reshape((-1,) + (1,) * (x_t.dim() - 1))
        next_xt = x_t + v_pred * (cur_t - next_t)

        return next_xt


class FMEulerSampler:
    def __init__(self, steps=None, t_pow_shift=2.0):
        self.default_steps = steps
        self.t_pow_shift = t_pow_shift

    @torch.compiler.disable(recursive=False)
    def sample(
        self,
        fn,
        fm_trainer,
        shape,
        steps=None,
        fn_kwargs=None,
        noise=None,
        device=None,
    ):
        if steps is None:
            if self.default_steps is None:
                raise ValueError("steps must be specified or default_steps must be set in the sampler")
            steps = self.default_steps

        if device is None:
            device = next(fn.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, steps + 1, device=device) ** self.t_pow_shift

        with torch.no_grad():
            for i in range(steps):
                t = t_steps[i].repeat(x_t.shape[0])
                v_pred = fm_trainer.get_prediction(
                    fn,
                    x_t=x_t,
                    t=t,
                    fn_kwargs=fn_kwargs,
                )
                x_t = fm_trainer.step(x_t, v_pred, t_steps[i], t_steps[i + 1])
        return x_t
