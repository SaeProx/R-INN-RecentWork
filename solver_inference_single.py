import torch
import numpy as np
import os
# Must import the class so Pickle can recognize the object
from arch import RINNModel
from data_load import MicrowaveDataset, load_names

# --- CONFIGURATION ---
# 1. Path to the Robust Model
MODEL_PATH = "rinn_model_robust.pth"

# 2. Path to TRAINING data (Required to get the Normalization Scaler)
# The model doesn't know "mm" or "dB", it only knows "Standard Deviations".
# We need the original data to recover the scaling factors.
TRAIN_DATA_X = "dataset_500DOE_aid/dataset_x.npy"
TRAIN_DATA_Y = "dataset_500DOE_aid/dataset_y.npy"

# 3. Path to the NEW Single Sample (The one you want to calculate NMSE for)
# Replace these with your actual filenames
SINGLE_SAMPLE_X = "dataset_1sample/dataset_x_w0.npy"
SINGLE_SAMPLE_Y = "dataset_1sample/dataset_y_w0.npy"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Running on {device}")

    # --- STEP 1: Load Normalization Stats ---
    print("[System] Loading training stats for normalization...")
    if not os.path.exists(TRAIN_DATA_X):
        print(f"Error: Training data not found at {TRAIN_DATA_X}")
        return

    # We load the dataset class to calculate Mean/Std
    stats_dataset = MicrowaveDataset(x_path=TRAIN_DATA_X, y_path=TRAIN_DATA_Y)
    x_names = load_names()

    # --- STEP 2: Load the Model ---
    print(f"[System] Loading {MODEL_PATH}...")
    try:
        # Load the full object directly
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
    except FileNotFoundError:
        print("Error: Model file not found.")
        return

    # --- STEP 3: Load the Single Sample ---
    print(f"[System] Loading target sample...")
    try:
        # Load the numpy files
        real_target_x = np.load(SINGLE_SAMPLE_X)
        real_target_y = np.load(SINGLE_SAMPLE_Y)

        # Ensure shape is [1, Dims] (Handle 1D array case)
        if len(real_target_x.shape) == 1:
            real_target_x = real_target_x.reshape(1, -1)
        if len(real_target_y.shape) == 1:
            real_target_y = real_target_y.reshape(1, -1)

    except FileNotFoundError:
        print("Error: Single sample files not found. Check the paths in CONFIG.")
        return

    # --- STEP 4: Normalize Input ---
    # Formula: (NewY - TrainMean) / TrainStd
    y_norm_np = (real_target_y - stats_dataset.y_mean) / stats_dataset.y_std
    y_tensor = torch.FloatTensor(y_norm_np).to(device)

    # --- STEP 5: Run Inverse Inference ---
    print("[System] Calculating Inverse Design...")
    with torch.no_grad():
        x_recon_padded, _ = model.inverse(y_tensor)

    # Extract just the geometry part
    pred_x_norm = x_recon_padded[:, :stats_dataset.x_dim].cpu().numpy()

    # --- STEP 6: Denormalize Output ---
    # Formula: (PredNorm * TrainStd) + TrainMean
    pred_x_real = (pred_x_norm * stats_dataset.x_std) + stats_dataset.x_mean

    # --- STEP 7: Report Results ---
    print("\n" + "=" * 65)
    print(f"INVERSE PREDICTION vs GROUND TRUTH")
    print("=" * 65)
    print(f"{'Parameter':<10} | {'Predicted':<15} | {'True Value':<15} | {'Diff':<10}")
    print("-" * 65)

    errors = []
    # Loop through parameters (H1, H2...)
    for i, name in enumerate(x_names):
        p_val = pred_x_real[0][i]
        t_val = real_target_x[0][i]
        diff = abs(p_val - t_val)
        errors.append((p_val - t_val) ** 2)  # Squared Error

        print(f"{name:<10} | {p_val:.5f}          | {t_val:.5f}          | {diff:.5f}")

    print("-" * 65)

    # Calculate NMSE (Normalized Mean Squared Error)
    # Note: This is usually calculated on the Normalized data to be unit-independent
    # But here we show physical MSE first.
    mse_physical = np.mean(errors)
    print(f"Physical MSE (mm^2):  {mse_physical:.6f}")

    # Calculate Normalized MSE (The number usually reported in papers)
    # This compares the 'Standard Deviations of error' rather than mm error
    true_x_norm = (real_target_x - stats_dataset.x_mean) / stats_dataset.x_std
    mse_normalized = np.mean((pred_x_norm - true_x_norm) ** 2)
    print(f"Normalized MSE (NMSE): {mse_normalized:.6f}")

    # --- STEP 8: Round Trip Check ---
    # Does this predicted geometry actually create the curve?
    x_pred_tensor = torch.zeros(1, model.input_dim).to(device)
    x_pred_tensor[:, :stats_dataset.x_dim] = torch.tensor(pred_x_norm).to(device)

    with torch.no_grad():
        z_verif, _, _ = model(x_pred_tensor)

    z_verif_np = z_verif.cpu().numpy()
    rt_error = np.mean((z_verif_np - y_norm_np) ** 2)

    print(f"Round-Trip Error:      {rt_error:.6f}")

    if rt_error < 0.1:
        print("\n>> CONCLUSION: SUCCESS.")
        print("   The predicted geometry is VALID (it generates the correct S-parameters).")
    else:
        print("\n>> CONCLUSION: WARNING.")
        print("   The predicted geometry might not match the target performance.")


if __name__ == "__main__":
    main()