from typing import List, Tuple

import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    """
    Implements Affine Coupling layer.
    Splits input features into two parts and uses one part to predict affine transformation parameters for the other.
    """
    
    def __init__(self, input_dim: int, x1_dim: int, hidden_dim: int) -> None:
        """
        Initialize AffineCoupling layer.
        
        Args:
            input_dim: Input feature dimension
            x1_dim: Dimension of conditional part x1
            hidden_dim: MLP hidden layer dimension
        """
        super(AffineCoupling, self).__init__()
        
        # Store input dimension and split information
        self.input_dim = input_dim
        self.x1_dim = x1_dim
        self.x2_dim = input_dim - x1_dim
        
        # Ensure x1_dim and x2_dim are both greater than 0
        if self.x1_dim <= 0 or self.x2_dim <= 0:
            raise ValueError(
                f"AffineCoupling split dimensions must both be greater than 0, "
                f"but got x1_dim={self.x1_dim}, x2_dim={self.x2_dim}"
            )
        
        # MLP for generating scale parameter
        self.scale_net = nn.Sequential(
            nn.Linear(self.x1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.x2_dim),
            nn.Tanh()  # Use tanh to limit scale range
        )
        
        # MLP for generating translate parameter
        self.translate_net = nn.Sequential(
            nn.Linear(self.x1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.x2_dim)
        )
    
    def _apply_forward_coupling(
        self, 
        h1: torch.Tensor, 
        h2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward affine coupling transformation.
        
        Args:
            h1: Conditional part, shape (batch_size, x1_dim)
            h2: Part to be transformed, shape (batch_size, x2_dim)
            
        Returns:
            h2_out: Transformed h2, shape (batch_size, x2_dim)
            scale: Scale parameter, shape (batch_size, x2_dim)
            log_det: Log determinant, shape (batch_size,)
        """
        # Generate scale and translate using conditional part through MLP
        scale = self.scale_net(h1)
        translate = self.translate_net(h1)
        
        # Apply affine transformation to the part to be transformed
        h2_out = h2 * torch.exp(scale) + translate
        
        # Calculate log_det
        log_det = scale.sum(dim=-1)
        
        return h2_out, scale, log_det
    
    def _apply_inverse_coupling(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply inverse affine coupling transformation.
        
        Args:
            z1: Conditional part, shape (batch_size, x1_dim)
            z2: Transformed part, shape (batch_size, x2_dim)
            
        Returns:
            z2_out: Recovered z2, shape (batch_size, x2_dim)
            scale: Scale parameter, shape (batch_size, x2_dim)
            log_det: Log determinant, shape (batch_size,)
        """
        # Generate scale and translate using conditional part through MLP
        scale = self.scale_net(z1)
        translate = self.translate_net(z1)
        
        # Apply inverse affine transformation to the transformed part
        z2_out = (z2 - translate) * torch.exp(-scale)
        
        # Calculate log_det for inverse process
        log_det = -scale.sum(dim=-1)
        
        return z2_out, scale, log_det
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AffineCoupling forward transformation.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            output: Transformed tensor, shape (batch_size, input_dim)
            log_det: Log determinant of Jacobian, shape (batch_size,)
        """
        # Check input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"AffineCoupling expects input dimension {self.input_dim}, "
                f"but got {x.shape[1]}"
            )
        
        # Split x into conditional part and part to be transformed
        h1 = x[:, :self.x1_dim]  # Conditional part
        h2 = x[:, self.x1_dim:]  # Part to be transformed
        
        # Apply forward coupling
        h2_out, _, log_det = self._apply_forward_coupling(h1, h2)
        
        # Concatenate conditional part and transformed part
        output = torch.cat([h1, h2_out], dim=-1)
        
        return output, log_det
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        AffineCoupling inverse transformation.
        
        Args:
            z: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            output: Inverse transformed tensor, shape (batch_size, input_dim)
            log_det: Log determinant of inverse process, shape (batch_size,)
        """
        # Check input dimension
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"AffineCoupling inverse expects input dimension {self.input_dim}, "
                f"but got {z.shape[1]}"
            )
        
        # Split z into conditional part and transformed part
        z1 = z[:, :self.x1_dim]  # Conditional part
        z2 = z[:, self.x1_dim:]  # Transformed part
        
        # Apply inverse coupling
        z2_out, _, log_det = self._apply_inverse_coupling(z1, z2)
        
        # Concatenate conditional part and recovered part
        output = torch.cat([z1, z2_out], dim=-1)
        
        return output, log_det


class Shuffle(nn.Module):
    """
    Shuffle layer: Permutes input feature order to increase model expressiveness.
    """
    
    def __init__(self, input_dim: int) -> None:
        """
        Initialize Shuffle layer.
        
        Args:
            input_dim: Input feature dimension
        """
        super(Shuffle, self).__init__()
        self.input_dim = input_dim
        
        # Pre-generate random permutation index perm
        self.perm = torch.randperm(input_dim)
        
        # Pre-generate inverse permutation inv_perm of perm
        self.inv_perm = torch.argsort(self.perm)
    
    def _apply_permutation(
        self, 
        x: torch.Tensor, 
        permutation: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply permutation to input tensor.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            permutation: Permutation indices
            
        Returns:
            output: Permuted tensor, shape (batch_size, input_dim)
        """
        return x[:, permutation]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shuffle forward transformation.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            output: Permuted tensor, shape (batch_size, input_dim)
            log_det: Log determinant of Jacobian, fixed at 0
        """
        # Check input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Shuffle expects input dimension {self.input_dim}, "
                f"but got {x.shape[1]}"
            )
        
        # Permute x features using perm
        output = self._apply_permutation(x, self.perm)
        
        # Return log_det=0 (permutation matrix determinant is ±1, log is approximately 0)
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        return output, log_det
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shuffle inverse transformation.
        
        Args:
            z: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            output: Restored order tensor, shape (batch_size, input_dim)
            log_det: Log determinant of Jacobian, fixed at 0
        """
        # Check input dimension
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"Shuffle inverse expects input dimension {self.input_dim}, "
                f"but got {z.shape[1]}"
            )
        
        # Restore z feature order using inv_perm
        output = self._apply_permutation(z, self.inv_perm)
        
        # Return log_det=0
        log_det = torch.zeros(z.shape[0], device=z.device)
        
        return output, log_det


class FlowCell(nn.Module):
    """
    FlowCell class: Contains one AffineCoupling and one Shuffle layer.
    Serves as the basic building block for FlowStage.
    """
    
    def __init__(self, input_dim: int, x1_dim: int, hidden_dim: int) -> None:
        """
        Initialize FlowCell.
        
        Args:
            input_dim: Input feature dimension
            x1_dim: Dimension of conditional part x1 in AffineCoupling
            hidden_dim: MLP hidden layer dimension
        """
        super(FlowCell, self).__init__()
        self.affine_coupling = AffineCoupling(input_dim, x1_dim, hidden_dim)
        self.shuffle = Shuffle(input_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FlowCell forward transformation.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            output: Transformed tensor, shape (batch_size, input_dim)
            log_det: Total log determinant, shape (batch_size,)
        """
        x, log_det1 = self.affine_coupling(x)
        x, log_det2 = self.shuffle(x)
        return x, log_det1 + log_det2
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FlowCell inverse transformation.
        
        Args:
            z: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            output: Inverse transformed tensor, shape (batch_size, input_dim)
            log_det: Total log determinant, shape (batch_size,)
        """
        z, log_det2 = self.shuffle.inverse(z)
        z, log_det1 = self.affine_coupling.inverse(z)
        return z, log_det1 + log_det2


class FlowStage(nn.Module):
    """
    FlowStage class: Implements the core logic of "hierarchical split + internal cycle" from RINN paper.
    First splits features, then performs multiple cycles on remaining features.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        z_part_dim: int, 
        h_prime_dim: int, 
        x1_dim: int, 
        hidden_dim: int, 
        num_cycles: int = 2
    ) -> None:
        """
        Initialize FlowStage.
        
        Args:
            input_dim: Input feature dimension
            z_part_dim: Dimension entering z output after split
            h_prime_dim: Dimension entering internal cycle after split
            x1_dim: Dimension of conditional part x1 in internal AffineCoupling
            hidden_dim: MLP hidden layer dimension
            num_cycles: Number of internal cycles
        """
        super(FlowStage, self).__init__()
        
        # Store dimension information
        self.input_dim = input_dim
        self.z_part_dim = z_part_dim
        self.h_prime_dim = h_prime_dim
        self.num_cycles = num_cycles
        
        # Ensure dimensions are valid
        if input_dim != z_part_dim + h_prime_dim:
            raise ValueError(
                f"FlowStage dimension mismatch: input_dim={input_dim}, "
                f"z_part_dim+h_prime_dim={z_part_dim+h_prime_dim}"
            )
        
        # Create FlowCell list for internal cycles
        self.cells = nn.ModuleList([
            FlowCell(self.h_prime_dim, x1_dim, hidden_dim) 
            for _ in range(num_cycles)
        ])
    
    def _split_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split input into z_part and h_prime.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            z_part: Current layer output z, shape (batch_size, z_part_dim)
            h_prime: Internal cycle input, shape (batch_size, h_prime_dim)
        """
        z_part = x[:, :self.z_part_dim]
        h_prime = x[:, self.z_part_dim:]
        return z_part, h_prime
    
    def _merge_output(
        self, 
        z_part: torch.Tensor, 
        h_prime: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge z_part and processed h_prime.
        
        Args:
            z_part: Current layer input z, shape (batch_size, z_part_dim)
            h_prime: Processed internal cycle output, shape (batch_size, h_prime_dim)
            
        Returns:
            x: Merged tensor, shape (batch_size, input_dim)
        """
        return torch.cat([z_part, h_prime], dim=-1)
    
    def _apply_internal_cycles(
        self, 
        h_prime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply internal cycles to h_prime.
        
        Args:
            h_prime: Input tensor, shape (batch_size, h_prime_dim)
            
        Returns:
            h_prime_out: Transformed tensor, shape (batch_size, h_prime_dim)
            log_det_total: Total log determinant, shape (batch_size,)
        """
        log_det_total = 0
        for cell in self.cells:
            h_prime, log_det = cell(h_prime)
            log_det_total += log_det
        return h_prime, log_det_total
    
    def _apply_inverse_cycles(
        self, 
        h_prime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse internal cycles to h_prime.
        
        Args:
            h_prime: Input tensor, shape (batch_size, h_prime_dim)
            
        Returns:
            h_prime_out: Inverse transformed tensor, shape (batch_size, h_prime_dim)
            log_det_total: Total log determinant, shape (batch_size,)
        """
        log_det_total = 0
        for cell in reversed(self.cells):
            h_prime, log_det = cell.inverse(h_prime)
            log_det_total += log_det
        return h_prime, log_det_total
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FlowStage forward transformation - split first, then cycle.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            z_part: Current layer output z, shape (batch_size, z_part_dim)
            h_prime: Output after internal cycles, shape (batch_size, h_prime_dim)
            log_det: Total log determinant of this layer, shape (batch_size,)
        """
        # Check input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"FlowStage expects input dimension {self.input_dim}, "
                f"but got {x.shape[1]}"
            )
        
        # Split: first z_part_dim parts as current layer output z, 
        # remaining h_prime_dim parts as internal cycle input h'
        z_part, h_prime = self._split_input(x)
        
        # Execute num_cycles internal cycles on h'
        h_prime, log_det_total = self._apply_internal_cycles(h_prime)
        
        return z_part, h_prime, log_det_total
    
    def inverse(
        self, 
        z_part: torch.Tensor, 
        h_prime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FlowStage inverse transformation - inverse cycle first, then merge.
        
        Args:
            z_part: Current layer input z, shape (batch_size, z_part_dim)
            h_prime: Internal cycle input, shape (batch_size, h_prime_dim)
            
        Returns:
            x: Inverse transformed tensor, shape (batch_size, input_dim)
            log_det: Total log determinant of inverse process, shape (batch_size,)
        """
        # Check input dimensions
        if z_part.shape[1] != self.z_part_dim:
            raise ValueError(
                f"FlowStage inverse expects z_part dimension {self.z_part_dim}, "
                f"but got {z_part.shape[1]}"
            )
        if h_prime.shape[1] != self.h_prime_dim:
            raise ValueError(
                f"FlowStage inverse expects h_prime dimension {self.h_prime_dim}, "
                f"but got {h_prime.shape[1]}"
            )
        
        # Execute num_cycles inverse internal cycles on h_prime in reverse order
        h_prime, log_det_total = self._apply_inverse_cycles(h_prime)
        
        # Merge z_part and processed h_prime
        x = self._merge_output(z_part, h_prime)
        
        return x, log_det_total


class RealNVP(nn.Module):
    """
    RealNVP main class: Implements complete normalizing flow model,
    conforming to "hierarchical split + internal cycle" design from RINN paper.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_stages: int = 4, 
        num_cycles_per_stage: int = 2, 
        ratio_toZ_after_flowstage: float = 0.5, 
        ratio_x1_x2_inAffine: float = 0.5
    ) -> None:
        """
        Initialize RealNVP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension of MLP in FlowCell
            num_stages: Number of flow stages
            num_cycles_per_stage: Number of internal cycles in each flow stage
            ratio_toZ_after_flowstage: Ratio entering z output after FlowStage split (0,1)
            ratio_x1_x2_inAffine: Ratio of conditional part x1 in AffineCoupling layer (0,1)
        """
        super(RealNVP, self).__init__()
        
        # Validate parameter validity
        if not (0 < ratio_toZ_after_flowstage < 1):
            raise ValueError(
                f"ratio_toZ_after_flowstage must be in (0,1), "
                f"but got {ratio_toZ_after_flowstage}"
            )
        if not (0 < ratio_x1_x2_inAffine < 1):
            raise ValueError(
                f"ratio_x1_x2_inAffine must be in (0,1), "
                f"but got {ratio_x1_x2_inAffine}"
            )
        
        # Store basic parameters
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.ratio_toZ_after_flowstage = ratio_toZ_after_flowstage
        self.ratio_x1_x2_inAffine = ratio_x1_x2_inAffine
        
        # Create flow stage list
        self.stages = nn.ModuleList()
        
        # Pre-compute and store dimension information for each stage, 
        # used for forward and inverse transformations
        self.stage_input_dims: List[int] = []      # Input dimension of each FlowStage
        self.z_part_dims: List[int] = []           # Dimension entering z output after FlowStage split
        self.h_prime_dims: List[int] = []          # Dimension entering internal cycle after FlowStage split
        self.cell_x1_dims: List[int] = []          # x1 dimension of AffineCoupling in each FlowCell
        
        # Current processing dimension
        current_dim = input_dim
        
        # Pre-compute dimensions for all layers
        for i in range(num_stages):
            # Ensure dimension is large enough for effective split
            if current_dim < 4:
                # Dimension too small, use minimum valid dimension
                current_dim = 4
                print(f"Warning: Stage {i+1} dimension adjusted to minimum valid value {current_dim}")
            
            # Calculate dimension split for current stage
            stage_input_dim = current_dim
            z_part_dim = max(1, int(stage_input_dim * ratio_toZ_after_flowstage))
            h_prime_dim = stage_input_dim - z_part_dim
            
            # Ensure split dimensions are both greater than 0
            if h_prime_dim <= 0:
                z_part_dim = stage_input_dim - 1
                h_prime_dim = 1
                print(
                    f"Warning: Stage {i+1} dimension split adjusted to "
                    f"z_part_dim={z_part_dim}, h_prime_dim={h_prime_dim}"
                )
            
            # Calculate x1 dimension of AffineCoupling in FlowCell
            x1_dim = max(1, int(h_prime_dim * ratio_x1_x2_inAffine))
            
            # Ensure x1_dim is valid
            if h_prime_dim - x1_dim <= 0:
                x1_dim = h_prime_dim - 1
                print(f"Warning: Stage {i+1} AffineCoupling split adjusted to x1_dim={x1_dim}")
            
            # Store dimension information
            self.stage_input_dims.append(stage_input_dim)
            self.z_part_dims.append(z_part_dim)
            self.h_prime_dims.append(h_prime_dim)
            self.cell_x1_dims.append(x1_dim)
            
            # Create FlowStage
            self.stages.append(FlowStage(
                stage_input_dim, 
                z_part_dim, 
                h_prime_dim, 
                x1_dim, 
                hidden_dim, 
                num_cycles_per_stage
            ))
            
            # Update current dimension (prepare for next stage)
            current_dim = h_prime_dim
    
    def _prepare_input(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare input by padding or truncating to match model input_dim.
        
        Args:
            x: Input data, shape (batch_size, data_dim)
            
        Returns:
            h: Prepared input, shape (batch_size, input_dim)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Ensure input dimension does not exceed model-defined input_dim
        if x.shape[1] > self.input_dim:
            # Truncate to input_dim
            h = x[:, :self.input_dim]
            print(
                f"Warning: Input data dimension {x.shape[1]} is greater than "
                f"model input_dim {self.input_dim}, truncated"
            )
        elif x.shape[1] < self.input_dim:
            # Zero pad to input_dim
            h = torch.zeros(batch_size, self.input_dim, device=device)
            h[:, :x.shape[1]] = x
        else:
            h = x
        
        return h
    
    def _prepare_output(
        self, 
        current_h: torch.Tensor, 
        original_data_dim: int
    ) -> torch.Tensor:
        """
        Prepare output by truncating or padding to match original input dimension.
        
        Args:
            current_h: Reconstructed tensor, shape (batch_size, current_dim)
            original_data_dim: Original input data dimension
            
        Returns:
            x_recon: Prepared output, shape (batch_size, original_data_dim)
        """
        batch_size = current_h.shape[0]
        device = current_h.device
        
        # Ensure final output dimension matches original input dimension
        if current_h.shape[1] > self.input_dim:
            # Truncate to original input dimension
            x_recon = current_h[:, :self.input_dim]
        elif current_h.shape[1] < self.input_dim:
            # Pad zeros to original input dimension
            padding = torch.zeros(
                batch_size, 
                self.input_dim - current_h.shape[1], 
                device=device, 
                dtype=current_h.dtype
            )
            x_recon = torch.cat([current_h, padding], dim=-1)
        else:
            x_recon = current_h
        
        return x_recon
    
    def _split_z_into_parts(
        self, 
        z: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Split z into output parts from each stage.
        
        Args:
            z: Transformed data, shape (batch_size, latent_dim)
            
        Returns:
            z_list: List of z parts from each stage
        """
        z_list = []
        start_idx = 0
        
        # Split output z_part from each stage
        for dim in self.z_part_dims:
            if start_idx + dim <= z.shape[1]:
                z_part = z[:, start_idx:start_idx + dim]
                z_list.append(z_part)
                start_idx += dim
            else:
                break
        
        # Add remaining features
        if start_idx < z.shape[1]:
            z_list.append(z[:, start_idx:])
        
        return z_list
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RealNVP overall forward transformation.
        
        Args:
            x: Input data, shape (batch_size, data_dim), where data_dim <= input_dim
            
        Returns:
            z: Transformed data, shape (batch_size, latent_dim)
            log_det_total: Total log determinant of Jacobian, shape (batch_size,)
        """
        # Prepare input
        h = self._prepare_input(x)
        
        # Initialize z_list and log_det_total
        z_list = []
        log_det_total = 0
        
        # Current processing features
        current_h = h
        
        # Iterate through each flow stage
        for i in range(self.num_stages):
            if i >= len(self.stages):
                break
            
            # Get current stage
            stage = self.stages[i]
            
            # Ensure current_h dimension matches
            if current_h.shape[1] != self.stage_input_dims[i]:
                raise ValueError(
                    f"Stage {i+1} expects input dimension {self.stage_input_dims[i]}, "
                    f"but got {current_h.shape[1]}"
                )
            
            # Execute forward transformation of current stage (split first, then cycle)
            z_part, current_h, log_det = stage(current_h)
            log_det_total += log_det
            
            # Add current stage output z
            z_list.append(z_part)
        
        # Add final remaining features as final z
        z_list.append(current_h)
        
        # Concatenate all z into one vector
        z = torch.cat(z_list, dim=-1)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RealNVP overall inverse transformation.
        
        Args:
            z: Transformed data, shape (batch_size, latent_dim)
            
        Returns:
            x_recon: Recovered input data, shape same as original input
            log_det_total: Total log determinant of inverse process, shape (batch_size,)
        """
        # Initialize log_det_total
        log_det_total = 0
        
        # Get batch size and device
        batch_size = z.shape[0]
        device = z.device
        
        # Split z into output parts from each stage
        z_list = self._split_z_into_parts(z)
        
        # Start from last feature as initial current_h
        if z_list:
            current_h = z_list.pop()
        else:
            # Special case handling
            current_h = torch.zeros(batch_size, 2, device=device, dtype=z.dtype)
        
        # Iterate through each flow stage in reverse order to reconstruct original features
        for i in reversed(range(min(len(self.stages), len(z_list)))):
            stage = self.stages[i]
            z_part = z_list.pop()
            
            # Execute inverse transformation of current stage (inverse cycle first, then merge)
            current_h, log_det = stage.inverse(z_part, current_h)
            log_det_total += log_det
        
        # Prepare output to match original input dimension
        x_recon = self._prepare_output(current_h, self.input_dim)
        
        return x_recon, log_det_total


# Reversibility verification example
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration parameters - test non-even dimensions
    input_dim = 13  # Target input dimension (non-even)
    batch_size = 32  # Batch size
    data_dim = 10  # Actual data dimension (less than input_dim, test zero padding)
    
    # Create model - use non-0.5 split ratio
    model = RealNVP(
        input_dim=input_dim,
        hidden_dim=64,
        num_stages=4,
        num_cycles_per_stage=2,
        ratio_toZ_after_flowstage=0.3,  # 30% enters z output
        ratio_x1_x2_inAffine=0.25      # 25% is x1 conditional part
    )
    
    # Generate random input (actual data dimension less than target input dimension)
    x = torch.randn(batch_size, data_dim)
    
    try:
        print(f"Original input x shape: {x.shape}")
        
        # Forward propagation
        z, log_det_forward = model(x)
        print(f"Forward propagation successful, z shape: {z.shape}")
        print(f"Forward log_det: {log_det_forward.mean().item():.4f}")
        
        # Inverse propagation
        x_recon, log_det_inverse = model.inverse(z)
        print(f"Inverse propagation successful, reconstructed x shape: {x_recon.shape}")
        print(f"Inverse log_det: {log_det_inverse.mean().item():.4f}")
        
        # Calculate MSE between recovered input and original input
        # Only compare original input data part, excluding zero padding part
        mse = torch.mean((x_recon[:, :data_dim] - x) ** 2)
        
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
