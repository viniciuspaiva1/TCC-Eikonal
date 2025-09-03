import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataGen import generateSpacedSourcesData
from architecture import Architecture
from efficient_kan.kan import KAN
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from metrics import score
from pathlib import Path

torch.manual_seed(42)

# Definições
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = ['x', 'z', 'source_x0', 'source_z0']
target = ['travel_time']
EPOCHS = 100
OUTPUT_DIR = Path("results")

# Define tamanhos de grade e espaçamentos correspondentes
grids_and_spacings = {
    (30, 30): [2, 3, 5],
    # (40, 40): [2, 3, 5],
    # (50, 50): [2, 5, 10],
    # (80, 80): [2, 5, 10],
    # (100, 100): [5, 10, 15]
}

def build_velocity_model(nPoints):
    xover = np.arange(nPoints[0])
    zover = np.arange(nPoints[1])
    v_line_z = 1.5 + zover
    v_line_x = np.tile(0.3 * xover, (len(zover), 1))
    v_depth = np.tile(v_line_z[:, np.newaxis], (1, len(xover))) + v_line_x
    return v_depth[:, :, np.newaxis]  # Shape: (X, Z, 1)

def physics_loss_fn(model, input_tensor, *args, **kwargs):
    x, z = input_tensor[:, 0:1], input_tensor[:, 1:2]
    source_x0, source_z0 = input_tensor[:, 2:3], input_tensor[:, 3:4]
    x.requires_grad_(True)
    z.requires_grad_(True)

    x_reconstructed = torch.cat([x, z, source_x0, source_z0], dim=1)
    yhat = model(x_reconstructed)
    travel_t = yhat[:, 0].view(-1, 1)

    v = 1.5 + 0.3 * x + z
    dtdx = torch.autograd.grad(travel_t, x, torch.ones_like(x), create_graph=True)[0]
    dtdz = torch.autograd.grad(travel_t, z, torch.ones_like(z), create_graph=True)[0]
    pde1 = dtdx**2 + dtdz**2 - (1/v)**2
    beta = 10.0
    neg_violation = torch.relu(-travel_t)
    loss_neg = torch.mean(neg_violation ** 2)
    return torch.mean(pde1 ** 2) + beta * loss_neg

def train_and_evaluate(df, output_path, spacing_label):
    # Split by unique sources
    initial_conditions = df[['source_x0', 'source_z0']].drop_duplicates()
    ic_train = initial_conditions.sample(frac=0.6, random_state=42)
    ic_val = initial_conditions.drop(ic_train.index).sample(frac=0.5, random_state=42)
    ic_test = initial_conditions.drop(ic_train.index.union(ic_val.index))

    df_train = df.merge(ic_train, on=['source_x0', 'source_z0'])
    df_val = df.merge(ic_val, on=['source_x0', 'source_z0'])
    df_test = df.merge(ic_test, on=['source_x0', 'source_z0'])

    # Dataset & Loader
    X_train = torch.tensor(df_train[features].values, dtype=torch.float32)
    y_train = torch.tensor(df_train[target].values, dtype=torch.float32)
    X_val = torch.tensor(df_val[features].values, dtype=torch.float32)
    y_val = torch.tensor(df_val[target].values, dtype=torch.float32)
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32)
    y_test = torch.tensor(df_test[target].values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=len(df_test))

    model = KAN([4, 8, 6, 4, 2, 1], grid_size=4, spline_order=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    loss_fn = torch.nn.MSELoss()

    arch = Architecture(
        model=model,
        loss_fn=loss_fn,
        physics_fn=physics_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_physics=1
    )

    arch.set_loaders(train_loader, val_loader)
    arch.train(EPOCHS, seed=42)

    # Save checkpoint
    ckpt_path = output_path / f"model_checkpoint.pt"
    arch.save_checkpoint(ckpt_path)

    # Predict & save plot
    y_pred = arch.predict(df_test[features].values)
    df_test['predicted_travel_time'] = y_pred

    for (sx, sz), group in df_test.groupby(['source_x0', 'source_z0']):
        grid_true = group.pivot(index='z', columns='x', values='travel_time')
        grid_pred = group.pivot(index='z', columns='x', values='predicted_travel_time')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(grid_true, cmap='terrain')
        axes[0].set_title("True - Pykonal")
        axes[1].imshow(grid_pred, cmap='terrain')
        axes[1].set_title("Predicted - PINN+KAN")

        for ax in axes:
            ax.plot(sx, sz, marker='*', color='red', markersize=10, label='Fonte')
            
        fig.suptitle(f"Grid {grid_true.shape[0]}x{grid_true.shape[1]} | Source ({sx},{sz})")
        plt.tight_layout()
        plt.savefig(output_path / f"comparison_source_{int(sx)}_{int(sz)}.png")
        plt.close()

    df_test.to_csv(output_path / f"test_predictions.csv", index=False)

# MAIN LOOP
for (grid_x, grid_z), spacings in grids_and_spacings.items():
    for spacing in spacings:
        print(f"📦 Grid {grid_x}x{grid_z} | Spacing {spacing}")
        shape = (grid_x, grid_z, 1)
        v_model = build_velocity_model((grid_x, grid_z))
        df_all = generateSpacedSourcesData(
            nPoints=shape,
            nodeIntervals=(1.0, 1.0, 1.0),
            velocityModel=v_model,
            spacings=[spacing]
        )[spacing]

        label = f"{grid_x}x{grid_z}_s{spacing}"
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_dir / "full_dataset.csv", index=False)
        train_and_evaluate(df_all, out_dir, label)
