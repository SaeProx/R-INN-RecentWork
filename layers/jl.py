import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations


class JLLayer(nn.Module):
    """
    Jacobian Linear Layer.
    
    Implements a learnable linear transformation layer with forward and inverse operations.
    The layer supports orthogonal initialization and optional weight normalization.
    """
    
    def __init__(
        self, 
        dim: int, 
        orthogonal_init: bool = True, 
        use_weight_norm: bool = False
    ) -> None:
        """
        Initialize JLLayer.
        
        Args:
            dim: Input and output dimension
            orthogonal_init: If True, initialize weights orthogonally. 
                           If False, use Xavier uniform initialization.
            use_weight_norm: If True, apply weight normalization to the linear layer.
        """
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim, bias=True)
        
        # Initialize weights and bias
        self._init_weights(orthogonal_init, use_weight_norm)
    
    def _init_weights(
        self, 
        orthogonal_init: bool, 
        use_weight_norm: bool
    ) -> None:
        """
        Initialize layer weights and optionally apply weight normalization.
        
        Args:
            orthogonal_init: If True, use orthogonal initialization.
                           If False, use Xavier uniform initialization.
            use_weight_norm: If True, apply weight normalization.
        """
        if orthogonal_init:
            nn.init.orthogonal_(self.linear.weight)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
        
        # Initialize bias to zeros
        nn.init.zeros_(self.linear.bias)
        
        # Optionally apply weight normalization
        if use_weight_norm:
            self.linear = parametrizations.weight_norm(
                self.linear, 
                name='weight', 
                dim=0
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward transformation: y = W @ x + b
        
        Args:
            x: Input tensor, shape (batch_size, dim)
            
        Returns:
            Output tensor, shape (batch_size, dim)
        """
        return self.linear(x)
    
    def inverse(self, x_prime: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation: x = W^T @ (x_prime - b)
        
        Args:
            x_prime: Output tensor to be inverse transformed, shape (batch_size, dim)
            
        Returns:
            Recovered input tensor, shape (batch_size, dim)
        """
        W = self.linear.weight  # Weight matrix used for forward pass
        b = self.linear.bias
        return F.linear(x_prime - b, W.T, bias=None)
    
    def get_ortho_loss(self) -> torch.Tensor:
        """
        Compute orthogonal regularization loss.
        
        Calculates how much the weight matrix W deviates from being orthogonal:
        ||W^T @ W - I||^2
        
        Returns:
            Scalar tensor representing the orthogonal loss
        """
        W = self.linear.weight
        rows, cols = W.shape
        
        # Compute W^T @ W
        WtW = torch.matmul(W.t(), W)
        
        # Generate identity matrix I (on same device as W)
        I = torch.eye(rows, device=W.device)
        
        # Compute squared Frobenius norm (sum of squared element differences)
        loss = torch.sum((WtW - I) ** 2)
        return loss
    
    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log absolute determinant of the Jacobian matrix.
        
        Calculates log|det(W)| and expands it to a batch vector.
        If the matrix is singular (det=0), replaces the corresponding logabsdet
        with a small value to avoid -inf.
        
        Args:
            x: Input tensor, shape (batch_size, dim)
            
        Returns:
            Log determinant tensor, shape (batch_size,)
        """
        W = self.linear.weight
        sign, logabsdet = torch.slogdet(W)  # Returns (sign, logabsdet)
        
        # If sign==0, the matrix is singular (det == 0)
        if torch.any(sign == 0):
            # Replace singular cases with small logabsdet value
            # Use -1e6 as penalty value to avoid -inf causing numerical issues
            logabsdet = torch.where(
                sign == 0, 
                torch.full_like(logabsdet, -1e6), 
                logabsdet
            )
        
        return logabsdet.expand(x.shape[0])
