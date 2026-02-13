import torch

__all__ = ["ActNorm1d", "ActNorm2d", "ActNorm3d"]


class ActNorm(torch.jit.ScriptModule):
    """
    Activation Normalization layer.
    Implements affine transformation: y = scale * x + bias
    """

    def __init__(self, num_features: int) -> None:
        """
        Initialize ActNorm layer.

        Args:
            num_features: Number of features/channels
        """
        super().__init__()
        self.num_features = num_features
        self.scale = torch.nn.Parameter(torch.zeros(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset_(self) -> "ActNorm":
        """
        Reset the initialization state.

        Returns:
            self for method chaining
        """
        self._initialized = torch.tensor(False)
        return self

    def _check_input_dim(self, x: torch.Tensor) -> None:
        """
        Check if input tensor has correct dimensions.
        Must be implemented by subclasses.

        Args:
            x: Input tensor

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError()  # pragma: no cover

    def _validate_feature_dimension(self, x: torch.Tensor) -> None:
        """
        Validate that the feature dimension matches num_features.

        Args:
            x: Input tensor

        Raises:
            ValueError: If feature dimension doesn't match num_features
        """
        if x.dim() == 2:
            # 1D case: (N, C)
            feature_dim = x.shape[1]
        elif x.dim() == 3:
            # 1D sequence case: (N, C, L)
            feature_dim = x.shape[1]
        elif x.dim() == 4:
            # 2D case: (N, C, H, W)
            feature_dim = x.shape[1]
        elif x.dim() == 5:
            # 3D case: (N, C, D, H, W)
            feature_dim = x.shape[1]
        else:
            # For unsupported dimensions, skip validation
            return

        if feature_dim != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} features, "
                f"but got {feature_dim} features"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward transformation: y = scale * x + bias

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        self._check_input_dim(x)
        self._validate_feature_dimension(x)

        if x.dim() > 2:
            x = x.transpose(1, -1)

        if not self._initialized:
            # Calculate mean and std, avoiding issues when batch size is 1
            flat_x = x.detach().reshape(-1, x.shape[-1])
            mean_x = flat_x.mean(0)
            std_x = flat_x.std(0, unbiased=False)

            # Avoid zero standard deviation
            std_x = torch.clamp(std_x, min=1e-6)

            self.scale.data = 1.0 / std_x
            self.bias.data = -self.scale * mean_x
            self._initialized = torch.tensor(True)

        x = self.scale * x + self.bias

        if x.dim() > 2:
            x = x.transpose(1, -1)

        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation: x = (y - bias) / scale

        Args:
            y: Output tensor (tensor to be inverse transformed)

        Returns:
            Inverse transformed tensor

        Raises:
            RuntimeError: If ActNorm layer has not been initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "ActNorm layer must be initialized before calling inverse(). "
                "Please run forward() with some data first."
            )

        self._check_input_dim(y)
        self._validate_feature_dimension(y)

        # Handle dimension conversion for multi-dimensional tensors
        if y.dim() > 2:
            y = y.transpose(1, -1)

        # Execute inverse affine transformation: x = (y - bias) / scale
        x = (y - self.bias) / self.scale

        # Restore original dimension order
        if x.dim() > 2:
            x = x.transpose(1, -1)

        return x

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate log determinant of Jacobian.

        For affine transformation y = scale * x + bias,
        the Jacobian matrix is diagonal with diagonal elements as scale.
        Therefore log|det(J)| = sum(log|scale|)

        Args:
            x: Input tensor

        Returns:
            Scalar tensor of log determinant of Jacobian
        """
        if not self._initialized:
            raise RuntimeError(
                "ActNorm layer must be initialized before calling log_det_jacobian(). "
                "Please run forward() with some data first."
            )

        self._check_input_dim(x)

        # Calculate batch size and spatial dimensions
        batch_size = x.shape[0]

        if x.dim() == 2:
            # 1D case: (N, C)
            spatial_dims = 1
        elif x.dim() == 3:
            # 1D sequence case: (N, C, L)
            spatial_dims = x.shape[2]
        elif x.dim() == 4:
            # 2D case: (N, C, H, W)
            spatial_dims = x.shape[2] * x.shape[3]
        elif x.dim() == 5:
            # 3D case: (N, C, D, H, W)
            spatial_dims = x.shape[2] * x.shape[3] * x.shape[4]
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        # log|det(J)| = spatial_dims * sum(log|scale|)
        log_det = spatial_dims * torch.sum(torch.log(torch.abs(self.scale)))

        # Return tensor of batch size, log_det is the same for each sample
        return log_det.expand(batch_size)


class ActNorm1d(ActNorm):
    """
    1D Activation Normalization layer.
    Supports inputs of shape (N, C) or (N, C, L).
    """

    def _check_input_dim(self, x: torch.Tensor) -> None:
        """
        Check if input has correct dimensions for 1D ActNorm.

        Args:
            x: Input tensor

        Raises:
            ValueError: If input is not 2D or 3D
        """
        if x.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")


class ActNorm2d(ActNorm):
    """
    2D Activation Normalization layer.
    Supports inputs of shape (N, C, H, W).
    """

    def _check_input_dim(self, x: torch.Tensor) -> None:
        """
        Check if input has correct dimensions for 2D ActNorm.

        Args:
            x: Input tensor

        Raises:
            ValueError: If input is not 4D
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")


class ActNorm3d(ActNorm):
    """
    3D Activation Normalization layer.
    Supports inputs of shape (N, C, D, H, W).
    """

    def _check_input_dim(self, x: torch.Tensor) -> None:
        """
        Check if input has correct dimensions for 3D ActNorm.

        Args:
            x: Input tensor

        Raises:
            ValueError: If input is not 5D
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (got {x.dim()}D input)")