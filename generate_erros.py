import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from architecture import Architecture
from utils.metrics import score
from pathlib import Path
from typing import List, Optional
from efficient_kan.kan import KAN
import torch
import torch.optim as optim

# Caminhos
RESULTS_DIR = Path("results")
features = ['x', 'z', 'source_x0', 'source_z0']
target = ['travel_time']

summary = []

def plot_mse_heatmap_grid(
    dataset: pd.DataFrame,
    model,
    features: List[str],
    target: str,
    source: Optional[str] = None,
    ncols: int = 2,
    figsize: tuple = (12, 8),
    save_path: Optional[Path] = None
):
    sources = dataset[['source_x0', 'source_z0']].drop_duplicates().values
    nplots = len(sources)
    nrows = (nplots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = np.array(axs).reshape(-1)

    for ax, (sx, sz) in zip(axs, sources):
        df_sub = dataset[(dataset['source_x0'] == sx) & (dataset['source_z0'] == sz)].copy()

        X = df_sub[features].values
        yhat = model.predict(X)
        y_true = df_sub[target].values

        df_sub['se'] = (yhat - y_true) ** 2

        mse_df = df_sub.groupby(['x', 'z'], as_index=False)['se'].mean()
        pivot = mse_df.pivot(index='z', columns='x', values='se')
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)

        x_vals = pivot.columns.values
        z_vals = pivot.index.values
        mse_grid = pivot.values

        im = ax.imshow(
            mse_grid,
            origin='upper',
            extent=[x_vals.min(), x_vals.max(), z_vals.max(), z_vals.min()],
            aspect='auto'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        title = f'MSE ({sx}, {sz})'
        if source:
            title += f' - {source}'
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='MSE')

    for unused_ax in axs[nplots:]:
        unused_ax.axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

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

# Loop principal
for grid_folder in sorted(RESULTS_DIR.iterdir()):
    if not grid_folder.is_dir():
        continue

    print(f"\n📂 Analisando: {grid_folder.name}")

    checkpoint_path = grid_folder / "model_checkpoint.pt"
    test_data_path = grid_folder / "test_predictions.csv"

    if not checkpoint_path.exists() or not test_data_path.exists():
        print(f"⚠️  Checkpoint ou CSV de teste não encontrado em {grid_folder.name}")
        continue

    # Carrega o modelo
    arch.load_checkpoint(checkpoint_path)

    # Plota e salva o loss curve
    fig = arch.plot_losses()
    fig.suptitle(f"Loss - {grid_folder.name}")
    fig.savefig(grid_folder / "loss_curve.png")
    plt.close(fig)

    # Carrega os dados de teste
    df_test = pd.read_csv(test_data_path)

    # Calcula os erros médios
    y_true = df_test[target].values
    y_pred = df_test['predicted_travel_time'].values.reshape(-1, 1)
    result_df = score(y_pred, y_true, labels=["Travel Time"])
    mae = result_df.loc["MAE", "Travel Time"]
    mse = result_df.loc["MSE", "Travel Time"]
    rmse = result_df.loc["RMSE", "Travel Time"]

    summary.append({
        "Grid_Spacing": grid_folder.name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    # Plota e salva o heatmap de MSE por fonte
    plot_mse_heatmap_grid(
        dataset=df_test,
        model=arch,
        features=features,
        target=target,
        source=grid_folder.name,
        save_path=grid_folder / "mse_heatmap.png"
    )

    # Adiciona ao resumo
    summary.append({
        "Grid_Spacing": grid_folder.name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    print(f"✅ MAE: {mae:.4f} | MSE: {mse:.6f} | RMSE: {rmse:.4f}")

# Salva o CSV com os erros de todos os grids
df_summary = pd.DataFrame(summary)
df_summary.to_csv(RESULTS_DIR / "summary_errors.csv", index=False)

print("\n✅ Resumo salvo em:", RESULTS_DIR / "summary_errors.csv")
