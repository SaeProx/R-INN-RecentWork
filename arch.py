from typing import Tuple

import torch
import torch.nn as nn

from layers import ActNorm1d,JLLayer,RealNVP
#from JL.jl import
#from realnvp.realnvp import


# Default configuration constants
DEFAULT_HIDDEN_DIM: int = 64
DEFAULT_NUM_STAGES: int = 4
DEFAULT_NUM_CYCLES_PER_STAGE: int = 2
DEFAULT_RATIO_TO_Z_AFTER_FLOWSTAGE: float = 0.3
DEFAULT_RATIO_X1_X2_IN_AFFINE: float = 0.25
DEFAULT_ORTHOGONAL_INIT: bool = True
DEFAULT_USE_WEIGHT_NORM: bool = False


class RINNBlock(nn.Module):
    """
    R-INN basic building block.
    
    Combines ActNorm → RealNVP → JL layers in sequence to implement
    reversible transformations and Jacobian determinant computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_stages: int = DEFAULT_NUM_STAGES,
        num_cycles_per_stage: int = DEFAULT_NUM_CYCLES_PER_STAGE,
        ratio_to_z_after_flowstage: float = DEFAULT_RATIO_TO_Z_AFTER_FLOWSTAGE,
        ratio_x1_x2_in_affine: float = DEFAULT_RATIO_X1_X2_IN_AFFINE,
        orthogonal_init: bool = DEFAULT_ORTHOGONAL_INIT,
        use_weight_norm: bool = DEFAULT_USE_WEIGHT_NORM
    ) -> None:
        """
        Initialize RINNBlock.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension of MLP in RealNVP
            num_stages: Number of flow stages in RealNVP
            num_cycles_per_stage: Number of internal cycles per flow stage
            ratio_to_z_after_flowstage: Ratio entering z output after FlowStage split
            ratio_x1_x2_in_affine: Ratio of conditional part x1 in AffineCoupling
            orthogonal_init: Whether to initialize JL layer weights orthogonally
            use_weight_norm: Whether to apply weight normalization to JL layer
        """
        super(RINNBlock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.num_cycles_per_stage = num_cycles_per_stage
        
        # Build model layers
        self._build_model_layers(
            input_dim,
            hidden_dim,
            num_stages,
            num_cycles_per_stage,
            ratio_to_z_after_flowstage,
            ratio_x1_x2_in_affine,
            orthogonal_init,
            use_weight_norm
        )
    
    def _build_model_layers(
        self,
        input_dim: int,
        hidden_dim: int,
        num_stages: int,
        num_cycles_per_stage: int,
        ratio_to_z_after_flowstage: float,
        ratio_x1_x2_in_affine: float,
        orthogonal_init: bool,
        use_weight_norm: bool
    ) -> None:
        """
        Build and initialize all model layers.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension of MLP in RealNVP
            num_stages: Number of flow stages in RealNVP
            num_cycles_per_stage: Number of internal cycles per flow stage
            ratio_to_z_after_flowstage: Ratio entering z output after FlowStage split
            ratio_x1_x2_in_affine: Ratio of conditional part x1 in AffineCoupling
            orthogonal_init: Whether to initialize JL layer weights orthogonally
            use_weight_norm: Whether to apply weight normalization to JL layer
        """
        # 1. ActNorm layer: Normalize input data
        self.actnorm = ActNorm1d(num_features=input_dim)
        
        # 2. RealNVP layer: Execute reversible flow transformation
        self.realnvp = RealNVP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            num_cycles_per_stage=num_cycles_per_stage,
            ratio_toZ_after_flowstage=ratio_to_z_after_flowstage,
            ratio_x1_x2_inAffine=ratio_x1_x2_in_affine
        )
        
        # 3. JL layer: Execute linear transformation, maintaining reversibility
        self.jl_layer = JLLayer(
            dim=input_dim,
            orthogonal_init=orthogonal_init,
            use_weight_norm=use_weight_norm
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RINNBlock forward transformation.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            z: Transformed tensor, shape (batch_size, input_dim)
            log_det_total: Total log determinant of Jacobian, shape (batch_size,)
            ortho_loss: Orthogonal regularization loss, scalar tensor
        """
        # Step 1: ActNorm transformation
        x = self.actnorm(x)
        
        # Step 2: RealNVP transformation
        z, log_det_realnvp = self.realnvp(x)
        
        # Step 3: JL layer transformation
        z = self.jl_layer(z)
        
        # Step 4: Compute Jacobian determinants
        log_det_jl = self.jl_layer.log_det_jacobian(z)
        log_det_actnorm = self.actnorm.log_det_jacobian(x)
        
        # Step 5: Sum all log determinants
        log_det_total = log_det_realnvp + log_det_jl + log_det_actnorm
        
        # Step 6: Get orthogonal loss for this layer
        ortho_loss = self.jl_layer.get_ortho_loss()
        
        return z, log_det_total, ortho_loss
    
    def inverse(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RINNBlock inverse transformation.
        
        Args:
            z: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            x: Inverse transformed tensor, shape (batch_size, input_dim)
            log_det_total: Total log determinant of inverse process, shape (batch_size,)
        """
        # Step 1: Inverse JL layer transformation
        x = self.jl_layer.inverse(z)
        
        # Step 2: Inverse RealNVP transformation
        x, log_det_realnvp = self.realnvp.inverse(x)
        
        # Step 3: Inverse ActNorm transformation
        x = self.actnorm.inverse(x)
        
        # Step 4: Compute inverse process Jacobian determinants (note sign change)
        log_det_actnorm = -self.actnorm.log_det_jacobian(x)
        log_det_jl = -self.jl_layer.log_det_jacobian(z)
        
        # Step 5: Sum all log determinants
        log_det_total = log_det_realnvp + log_det_jl + log_det_actnorm
        
        return x, log_det_total


class RINNModel(nn.Module):
    """
    Complete R-INN model.
    
    Composed of multiple RINNBlock instances connected sequentially.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_blocks: int = 3,
        num_stages: int = DEFAULT_NUM_STAGES,
        num_cycles_per_stage: int = DEFAULT_NUM_CYCLES_PER_STAGE,
        ratio_to_z_after_flowstage: float = DEFAULT_RATIO_TO_Z_AFTER_FLOWSTAGE,
        ratio_x1_x2_in_affine: float = DEFAULT_RATIO_X1_X2_IN_AFFINE,
        orthogonal_init: bool = DEFAULT_ORTHOGONAL_INIT,
        use_weight_norm: bool = DEFAULT_USE_WEIGHT_NORM
    ) -> None:
        """
        Initialize RINNModel.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension of MLP in RealNVP
            num_blocks: Number of RINNBlock instances
            num_stages: Number of flow stages in each RINNBlock's RealNVP
            num_cycles_per_stage: Number of internal cycles per flow stage
            ratio_to_z_after_flowstage: Ratio entering z output after FlowStage split
            ratio_x1_x2_in_affine: Ratio of conditional part x1 in AffineCoupling
            orthogonal_init: Whether to initialize JL layer weights orthogonally
            use_weight_norm: Whether to apply weight normalization to JL layer
        """
        super(RINNModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.num_cycles_per_stage = num_cycles_per_stage
        
        # Build model layers
        self._build_model_layers(
            input_dim,
            hidden_dim,
            num_blocks,
            num_stages,
            num_cycles_per_stage,
            ratio_to_z_after_flowstage,
            ratio_x1_x2_in_affine,
            orthogonal_init,
            use_weight_norm
        )
    
    def _build_model_layers(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_stages: int,
        num_cycles_per_stage: int,
        ratio_to_z_after_flowstage: float,
        ratio_x1_x2_in_affine: float,
        orthogonal_init: bool,
        use_weight_norm: bool
    ) -> None:
        """
        Build and initialize all RINNBlock layers.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension of MLP in RealNVP
            num_blocks: Number of RINNBlock instances to create
            num_stages: Number of flow stages in each RINNBlock's RealNVP
            num_cycles_per_stage: Number of internal cycles per flow stage
            ratio_to_z_after_flowstage: Ratio entering z output after FlowStage split
            ratio_x1_x2_in_affine: Ratio of conditional part x1 in AffineCoupling
            orthogonal_init: Whether to initialize JL layer weights orthogonally
            use_weight_norm: Whether to apply weight normalization to JL layer
        """
        self.blocks = nn.ModuleList([
            RINNBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_stages=num_stages,
                num_cycles_per_stage=num_cycles_per_stage,
                ratio_to_z_after_flowstage=ratio_to_z_after_flowstage,
                ratio_x1_x2_in_affine=ratio_x1_x2_in_affine,
                orthogonal_init=orthogonal_init,
                use_weight_norm=use_weight_norm
            ) for _ in range(num_blocks)
        ])
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RINNModel forward transformation.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            z: Transformed tensor, shape (batch_size, input_dim)
            log_det_total: Total log determinant of Jacobian, shape (batch_size,)
            ortho_loss_total: Total orthogonal regularization loss, scalar tensor
            
        Raises:
            ValueError: If input dimension doesn't match model's input_dim
        """
        # Validate input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Model expects input dimension {self.input_dim}, "
                f"but got {x.shape[1]}"
            )
        
        # Initialize accumulation variables
        z = x
        log_det_total = 0
        ortho_loss_total = 0
        
        # Forward pass through each RINNBlock
        for block in self.blocks:
            z, log_det, ortho_loss = block(z)
            log_det_total += log_det
            ortho_loss_total += ortho_loss
        
        return z, log_det_total, ortho_loss_total
    
    def inverse(
        self, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RINNModel inverse transformation.
        
        Args:
            z: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            x_recon: Recovered input tensor, shape (batch_size, input_dim)
            log_det_total: Total log determinant of inverse process, shape (batch_size,)
            
        Raises:
            ValueError: If input dimension doesn't match model's input_dim
        """
        # Validate input dimension
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"Model inverse expects input dimension {self.input_dim}, "
                f"but got {z.shape[1]}"
            )
        
        # Initialize accumulation variables
        x_recon = z
        log_det_total = 0
        
        # Inverse pass through each RINNBlock (in reverse order)
        for block in reversed(self.blocks):
            x_recon, log_det = block.inverse(x_recon)
            log_det_total += log_det
        
        return x_recon, log_det_total


# Test code: Verify reversibility
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration parameters: Use 3 reversible blocks and 20-dimensional input/output
    input_dim = 20
    batch_size = 32
    num_blocks = 3  # 3 reversible blocks
    
    # Create RINN model
    model = RINNModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_blocks=num_blocks,
        num_stages=4,
        num_cycles_per_stage=2
    )
    
    # Generate random input
    x = torch.randn(batch_size, input_dim)
    
    try:
        print(f"Original input x shape: {x.shape}")
        print(f"R-INN model with {num_blocks} reversible blocks")
        
        # Forward propagation
        z, log_det_forward, ortho_loss = model(x)
        print(f"Forward propagation successful, z shape: {z.shape}")
        print(f"Forward log_det: {log_det_forward.mean().item():.4f}")
        print(f"Orthogonal loss: {ortho_loss.item():.4f}")
        
        # Inverse propagation
        x_recon, log_det_inverse = model.inverse(z)
        print(f"Inverse propagation successful, reconstructed x shape: {x_recon.shape}")
        print(f"Inverse log_det: {log_det_inverse.mean().item():.4f}")
        
        # Calculate MSE between recovered input and original input
        mse = torch.mean((x_recon - x) ** 2)
        
        # Print results
        print(f"Reversibility verification: MSE={mse.item():.10f}")
        if mse < 1e-5:
            print("✓ Reversibility verification passed! Reconstruction error less than 1e-5")
        else:
            print("⚠ Reversibility verification failed, reconstruction error is large")
    except Exception as e:
        print(f"Runtime error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
