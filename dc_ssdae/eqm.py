import torch

class GammaSamplerLogitNormal:
    def __init__(self):
            pass

    def __call__(self, batch_size, device):
        return torch.rand(batch_size, device=device)


class EquilibriumMatchingTrainer:
    def __init__(
        self,
        *,
        c_type: str = 'trunc',  # 'linear', 'trunc', 'piece'
        c_args={'grad_scale': 1.0, 'a': 0.8, 'b': 1.4}
    ):
        self.prediction_type = None
        self.gamma_sampler = GammaSamplerLogitNormal()
        self.c_type = c_type
        self.c_args = c_args

    def add_noise(self, x, gamma, s, noise=None):
        # gamma=1.0 -> no noise ; gamma=0.0 -> full noise
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_g = (1.0 - gamma).view(*s) * noise + gamma.view(*s) * x
        return x_g, noise
    
    def c(self, gamma):
            if self.c_type == 'linear':
                grad_scale = self.c_args.get('grad_scale', 4)
                return grad_scale * (1.0 - gamma)
            elif self.c_type == 'trunc':
                a = self.c_args.get('a', 0.8)
                grad_scale = self.c_args.get('grad_scale', 4)
                return grad_scale * torch.where(gamma > a, (1.0 - gamma) / (1.0 - a), torch.tensor(1.0, device=gamma.device))
            elif self.c_type == 'piece':
                a = self.c_args.get('a', 0.8)
                b = self.c_args.get('b', 1.4)
                grad_scale = self.c_args.get('grad_scale', 4)
                return grad_scale * torch.where(gamma > a, (1.0 - gamma) / (1.0 - a), b - (b - 1.0) * gamma / a)
            else:
                raise ValueError(f"Unknown c_type: {self.c_type}")

    def loss(self, fn, x, fn_kwargs=None, noise=None):
        if fn_kwargs is None:
            fn_kwargs = {}

        gamma = self.gamma_sampler(x.shape[0], device=x.device)
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_g, noise = self.add_noise(x, gamma, s, noise=noise)

        grad_pred = fn(x_g, **fn_kwargs)
        target = (noise - x) * self.c(gamma).view(*s)

        loss = ((grad_pred.float() - target.float()) ** 2).mean()
        return loss, (x_g, noise, 1.0 - gamma, grad_pred)

    def get_prediction(
        self,
        fn,
        x_g,
        fn_kwargs=None,
    ):
        return fn(x_g, **(fn_kwargs or {}))

    def step(self, x_g, grad_pred, step_size, grad_sample=False):
        if not isinstance(grad_pred, torch.Tensor):
            grad_pred = torch.tensor(grad_pred, device=x_g.device)
        
        step_size = step_size.reshape((-1,) + (1,) * (x_g.dim() - 1))
        if grad_sample:   # standard SGD step
            next_xt = x_g - grad_pred * step_size
        else:   # Used to predict x0 directly, convert to the same as Flow Matching
            gamma = 1.0 - step_size
            next_xt = x_g - grad_pred / self.c(gamma) * step_size

        return next_xt


class EqMEulerSampler:
    def __init__(self, steps=None, **kwargs):
        self.default_steps = steps

    @torch.compiler.disable(recursive=False)
    def sample(
        self,
        fn,
        eqm_trainer,
        shape,
        step_size=0.125,
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
        x_g = torch.randn(shape, device=device) if noise is None else noise

        with torch.no_grad():
            for i in range(steps):
                grad_pred = eqm_trainer.get_prediction(
                    fn,
                    x_g=x_g,
                    fn_kwargs=fn_kwargs,
                )
                x_g = eqm_trainer.step(x_g, grad_pred, step_size, grad_sample=True)
        return x_g
