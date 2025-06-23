import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
features = ['x', 'z', 'source_x0', 'source_z0']
target = ['travel_time']

for grid_folder in sorted(RESULTS_DIR.iterdir()):
    if not grid_folder.is_dir():
        continue

    test_data_path = grid_folder / "test_predictions.csv"
    if not test_data_path.exists():
        print(f"⚠️  Test data não encontrado em {grid_folder.name}")
        continue

    print(f"📂 Processando erros em: {grid_folder.name}")

    # Carrega o CSV
    df = pd.read_csv(test_data_path)

    # Calcula o erro ponto a ponto
    df['erro'] = df['travel_time'] - df['predicted_travel_time']

    # Para cada fonte (source_x0, source_z0), faz o heatmap do erro espacial
    for (sx, sz), group in df.groupby(['source_x0', 'source_z0']):
        grid_error = group.pivot(index='z', columns='x', values='erro')

        plt.figure(figsize=(8, 6))
        plt.imshow(grid_error, cmap='RdBu', interpolation='none')
        plt.colorbar(label='Erro (Original - Predito)')
        plt.title(f"Erro Grid: {grid_folder.name} | Fonte: ({int(sx)}, {int(sz)})")
        plt.xlabel('x')
        plt.ylabel('z')

        output_path = grid_folder / f"error_source_{int(sx)}_{int(sz)}.png"
        plt.savefig(output_path)
        plt.close()

    print(f"✅ Erros plotados e salvos para {grid_folder.name}")

print("\n✅ Finalizado!")