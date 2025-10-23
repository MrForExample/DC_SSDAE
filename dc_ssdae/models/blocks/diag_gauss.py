import torch


def soft_clamp(x, max_val, softness=2.0):
    """
    Applies a soft clamping function to the input tensor x.
    softness: Controls the steepness of the transition. Higher values mean a harder clamp.
    Used to avoid SDPBackend.EFFICIENT_ATTENTION RuntimeError: Function 'ScaledDotProductEfficientAttentionBackward0' returned nan values in its 0th output. indicated by large KL loss values.
    """
    # Scale x to be in a range suitable for tanh
    scaled_x = x / max_val

    # Apply tanh for a smooth, bounded output
    soft_clamped_scaled = torch.tanh(scaled_x * softness)
    
    # Rescale and shift back to the desired range
    soft_clamped = soft_clamped_scaled * max_val
    
    return soft_clamped

class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.mean = soft_clamp(self.mean, max_val=10.0, softness=2.0)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        with torch.autocast("cuda", enabled=False):
            self.std = torch.exp(0.5 * self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.std)

    def kl(self) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            with torch.autocast("cuda", enabled=False):
                mean = self.mean.to(torch.float32)
                logvar = self.logvar.to(torch.float32)
                var = torch.exp(self.logvar)
                return 0.5 * torch.sum(
                    torch.pow(mean, 2) + var - 1.0 - logvar,
                    dim=[1, 2, 3],
                )
