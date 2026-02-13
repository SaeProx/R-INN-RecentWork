import torch
import torch.optim as optim
import numpy as np
import copy
import random
import os
from arch import RINNModel
from data_load import MicrowaveDataset, load_names
from torch.optim.lr_scheduler import StepLR

# --- CONFIGURATION ---
BATCH_SIZE = 32
NUM_EPOCHS = 3000
HIDDEN_DIM = 128
NUM_BLOCKS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Filename
BEST_MODEL_PATH = "rinn_model_robust.pth"


def calculate_loss(model, x, y, w_x=50.0, w_y=50.0, w_z=0.0001, w_ortho=10.0):
    batch_size = x.shape[0]
    device = x.device
    model_dim = model.input_dim

    x_padded = torch.zeros(batch_size, model_dim).to(device)
    real_x_dim = x.shape[1]
    x_padded[:, :real_x_dim] = x

    z, log_det_forward, ortho_loss = model(x_padded)
    x_recon_full, _ = model.inverse(z)
    x_recon = x_recon_full[:, :real_x_dim]

    Ly = torch.mean((z - y) ** 2)
    Lx = torch.mean((x_recon - x) ** 2)
    L_jacobian = -torch.mean(log_det_forward)

    total_loss = (w_x * Lx) + (w_y * Ly) + (w_z * L_jacobian) + (w_ortho * ortho_loss)
    return total_loss


def main_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================
    # PART 1: TRAINING
    # ==========================================

    # 1. Load Data
    data_manager = MicrowaveDataset()
    train_loader = data_manager.get_train_loader(batch_size=BATCH_SIZE)

    real_x_dim = data_manager.x_dim
    real_y_dim = data_manager.y_dim
    model_dim = max(real_x_dim, real_y_dim)

    # 2. Initialize Model
    model = RINNModel(input_dim=model_dim, hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS, num_stages=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

    best_mse = float('inf')

    print("\n>>> STARTING TRAINING <<<")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = calculate_loss(model, batch_x, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # VALIDATION (Every 50 epochs)
        if (epoch + 1) % 50 == 0:
            model.eval()
            test_x, test_y = data_manager.get_test_data()
            test_x, test_y = test_x.to(device), test_y.to(device)

            with torch.no_grad():
                x_recon_padded, _ = model.inverse(test_y)
                x_pred = x_recon_padded[:, :real_x_dim]
                curr_mse = torch.mean((x_pred - test_x) ** 2).item()

            print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.4f} | Test MSE: {curr_mse:.6f}")

            # SAVE IF BEST
            if curr_mse < best_mse:
                best_mse = curr_mse
                # NUCLEAR OPTION: Save Full Object (Not just state_dict)
                # This captures the exact state of ActNorm layers
                torch.save(model, BEST_MODEL_PATH)
                print(f"   >>> SAVED TO DISK (MSE: {best_mse:.6f})")

    # ==========================================
    # PART 2: VERIFICATION
    # ==========================================
    print("\n>>> VERIFYING SAVED FILE <<<")

    # 1. Load the Full Object
    print(f"Loading {BEST_MODEL_PATH}...")
    # FIX: weights_only=False allows loading the custom class structure
    final_model = torch.load(BEST_MODEL_PATH, weights_only=False)
    final_model.eval()

    # 2. Test on Random Sample
    idx = random.randint(0, data_manager.n_samples - 1)
    true_y_norm = data_manager.y_data[idx].unsqueeze(0).to(device)
    true_x_norm = data_manager.x_data[idx].numpy()

    with torch.no_grad():
        x_recon_padded, _ = final_model.inverse(true_y_norm)

    pred_x_norm = x_recon_padded[0, :real_x_dim].cpu().numpy()

    # Calculate Error
    mse_check = np.mean((pred_x_norm - true_x_norm) ** 2)
    print(f"Verification MSE: {mse_check:.6f}")

    if mse_check < 0.1:
        print("SUCCESS! The model matches the training performance.")
    else:
        print("FAILURE. Still having loading issues.")


if __name__ == "__main__":
    main_pipeline()