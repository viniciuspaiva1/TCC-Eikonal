import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
features = ['x', 'z', 'source_x0', 'source_z0']
target = ['travel_time']

# --- ETAPA 1: Encontrar o VMAX global em todos os arquivos ---
print("🔎 Etapa 1: Calculando o erro máximo global em todos os arquivos...")
global_max_error = 0.0

for grid_folder in sorted(RESULTS_DIR.iterdir()):
    if not grid_folder.is_dir():
        continue
    
    test_data_path = grid_folder / "test_predictions.csv"
    if not test_data_path.exists():
        continue

    # Carrega o CSV temporariamente para encontrar o erro máximo
    df = pd.read_csv(test_data_path)
    df['erro'] = df['travel_time'] - df['predicted_travel_time']
    
    # Encontra o erro máximo absoluto neste arquivo
    max_error_in_file = np.max(np.abs(df['erro']))
    
    # Atualiza o erro máximo global se o do arquivo atual for maior
    if max_error_in_file > global_max_error:
        global_max_error = max_error_in_file
        print(f"   -> Novo erro máximo global encontrado: {global_max_error:.4f} em {grid_folder.name}")

print(f"\n✅ Erro máximo global definido para a escala: {global_max_error:.4f}\n")


# --- ETAPA 2: Gerar todos os gráficos com a escala global ---
print("🎨 Etapa 2: Gerando os gráficos com a escala de cor unificada...")
for grid_folder in sorted(RESULTS_DIR.iterdir()):
    if not grid_folder.is_dir():
        continue

    test_data_path = grid_folder / "test_predictions.csv"
    if not test_data_path.exists():
        print(f"⚠️  Test data não encontrado em {grid_folder.name} (etapa de plotagem)")
        continue

    print(f"📂 Processando e plotando erros em: {grid_folder.name}")

    # Carrega o CSV para plotagem
    df = pd.read_csv(test_data_path)

    # Calcula o erro ponto a ponto
    df['erro'] = df['travel_time'] - df['predicted_travel_time']

    # Para cada fonte (source_x0, source_z0), faz o heatmap do erro espacial
    for (sx, sz), group in df.groupby(['source_x0', 'source_z0']):
        grid_error = group.pivot(index='z', columns='x', values='erro')

        plt.figure(figsize=(8, 6))
        # --- ALTERAÇÃO PRINCIPAL ---
        # Usa o global_max_error (calculado na Etapa 1) para a escala de cores
        plt.imshow(grid_error, cmap='RdBu', vmax=global_max_error, vmin=-global_max_error, interpolation='none')
        # -------------------------
        
        plt.colorbar(label='Erro (Original - Predito)')
        plt.title(f"Erro Grid: {grid_folder.name} | Fonte: ({int(sx)}, {int(sz)})")
        plt.xlabel('x')
        plt.ylabel('z')

        output_path = grid_folder / f"error_source_{int(sx)}_{int(sz)}.png"
        plt.savefig(output_path)
        plt.close()

    print(f"✅ Erros plotados e salvos para {grid_folder.name}")

print("\n✅ Finalizado!")