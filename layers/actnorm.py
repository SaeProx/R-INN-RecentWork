import torch
import torch.nn as nn

__all__ = ["ActNorm1d", "ActNorm2d", "ActNorm3d"]


class ActNorm(nn.Module):  # <--- CHANGED from torch.jit.ScriptModule
    """
    Activation Normalization layer (Standard nn.Module version).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.scale = nn.Parameter(torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Register as a buffer so it saves to disk
        self.register_buffer("_initialized", torch.tensor(False))

    def reset_(self) -> "ActNorm":
        self._initialized.fill_(False)
        return self

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()

    def _validate_feature_dimension(self, x: torch.Tensor) -> None:
        if x.dim() == 2:
            feature_dim = x.shape[1]
        elif x.dim() == 3:
            feature_dim = x.shape[1]
        elif x.dim() == 4:
            feature_dim = x.shape[1]
        elif x.dim() == 5:
            feature_dim = x.shape[1]
        else:
            return

        if feature_dim != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        self._validate_feature_dimension(x)

        if x.dim() > 2: x = x.transpose(1, -1)

        if not self._initialized:
            # Data-dependent initialization
            flat_x = x.detach().reshape(-1, x.shape[-1])
            mean_x = flat_x.mean(0)
            std_x = flat_x.std(0, unbiased=False)
            std_x = torch.clamp(std_x, min=1e-6)

            self.scale.data = 1.0 / std_x
            self.bias.data = -self.scale * mean_x
            self._initialized.fill_(True)

        y = self.scale * x + self.bias

        if x.dim() > 2: y = y.transpose(1, -1)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # If loading failed, force initialization to avoid crashing
        # But this implies identity transform if weights are zero.
        # Ideally, weights are loaded correctly.
        if not self._initialized:
            # Fallback: assume initialized if we are running inverse (inference mode)
            pass

        self._check_input_dim(y)
        self._validate_feature_dimension(y)

        if y.dim() > 2: y = y.transpose(1, -1)

        x = (y - self.bias) / self.scale

        if x.dim() > 2: x = x.transpose(1, -1)
        return x

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)
        batch_size = x.shape[0]

        if x.dim() == 2:
            spatial_dims = 1
        elif x.dim() == 3:
            spatial_dims = x.shape[2]
        elif x.dim() == 4:
            spatial_dims = x.shape[2] * x.shape[3]
        elif x.dim() == 5:
            spatial_dims = x.shape[2] * x.shape[3] * x.shape[4]
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        log_det = spatial_dims * torch.sum(torch.log(torch.abs(self.scale)))
        return log_det.expand(batch_size)


class ActNorm1d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D)")


class ActNorm2d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D)")


class ActNorm3d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (got {x.dim()}D)")
