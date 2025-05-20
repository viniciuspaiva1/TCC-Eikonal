# %%
%load_ext autoreload
%autoreload 2

import DataGen as dtg
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from efficient_kan.kan import KAN
from torch.utils.data import TensorDataset, DataLoader, Sampler

import torch.optim as optim
from architecture import Architecture
from metrics import score
SEED = 42

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using '{DEVICE}' device.")

# %% [markdown]
# ## Obtenção do conjunto de dados

# %% [markdown]
# ### Definindo o espaço de velocidade
# 

# %%
X_POINTS = 3.0
Z_POINTS = 3.0
Y_POINTS = 0.1

X_DISTANCE = 0.1
Y_DISTANCE = 0.1
Z_DISTANCE = 0.1

dg_over = 100 / 1000

SHAPE = (int(X_POINTS/dg_over), int(Z_POINTS/dg_over), int(Y_POINTS/dg_over))


# %%
X_POINTS = 30
Z_POINTS = 30
Y_POINTS = 1

X_DISTANCE = 1
Y_DISTANCE = 1
Z_DISTANCE = 1

dg_over = 100 / 100

SHAPE = (int(X_POINTS/dg_over), int(Z_POINTS/dg_over), int(Y_POINTS/dg_over))

# %%
# alterar os limites do arange

xover = np.arange(0, 2900 / 100 + dg_over, dg_over)
zover = np.arange(0, 2900 / 100 + dg_over, dg_over)

v_line_z = 1.5 + zover
v_line_x = np.tile(0.3 * xover, (len(zover), 1))
v_depth = np.tile(v_line_z[:, np.newaxis], (1, len(xover))) + v_line_x

v_depth = v_depth.reshape(SHAPE) # necessário adicionar a terceira dimensão para o modelo do Solver

# %%
df = dtg.generateMultipleData(
    nPoints = SHAPE, 
    nodeDistances = (X_DISTANCE, Y_DISTANCE, Z_DISTANCE),
    velocityModel = v_depth,
    sourceOriginInterval= [[0,0],[7,7]])

# %%
len(df)

# %%
df[['x', 'z', 'source_x0', 'source_z0']] = df[['x', 'z', 'source_x0', 'source_z0']]
df

# %%
features = ['x', 'z', 'source_x0', 'source_z0']
target = ['travel_time']

# %% [markdown]
# ### Separando o conjunto em Treinamento, Validação e Teste

# %%
initial_conditions = df[['source_x0', 'source_z0']].drop_duplicates().reset_index(drop=True)
aux = initial_conditions.copy()

ic_train = aux.sample(frac=0.6, random_state=SEED)
ic_val = aux.drop(ic_train.index).sample(frac=0.5, random_state=SEED)
ic_test = aux.drop(list(ic_train.index) + list(ic_val.index))

print("Train Initial Conditions:", len(ic_train))
print("Validation Initial Conditions:", len(ic_val))
print("Test Initial Conditions:", len(ic_test))

# %%
df_train = df.merge(ic_train, on=['source_x0', 'source_z0'], how='inner')
df_val = df.merge(ic_val, on=['source_x0', 'source_z0'], how='inner')
df_test = df.merge(ic_test, on=['source_x0', 'source_z0'], how='inner')

print("Train Size:", len(df_train))
print("Validation Size:", len(df_val))
print("Test Size:", len(df_test))

# %%
df
# .loc[(df['source_x0'] == 0) & (df['source_z0'] == 0)]

# %%
from itertools import cycle
import random


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, t_index=3, shuffle=True):
        """
    A sampler that generates balanced batches from a dataset, ensuring that each batch contains
    an equal number of samples with a specific target value.
    Args:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): The number of samples per batch.
        t_index (int, optional): The index of the target value in the dataset samples. Default is 4.
        shuffle (bool, optional): Whether to shuffle the batch indices. Default is True.
        
    Attributes:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): The number of samples per batch.
        t_index (int): The index of the target value in the dataset samples.
        shuffle (bool): Whether to shuffle the batch indices.
        t_zero_indices (list): Indices of samples with target value equal to 0.
        non_t_zero_indices (list): Indices of samples with target value not equal to 0.
        
    Methods:
        __iter__(): Generates balanced batches of indices.
        __len__(): Returns the number of batches.
    """
        if batch_size % 2 != 0 or batch_size <= 0:
            raise ValueError("batch_size deve ser um número inteiro positivo e par.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.t_index = t_index
        self.shuffle = shuffle

        # Dividir o dataset em duas partes: com t=0 e sem t=0
        self.t_zero_indices = []
        self.non_t_zero_indices = []

        for i in range(len(self.dataset)):
            data = self.dataset[i]
            try:
                t_sample = data[0][self.t_index]
            except (IndexError, TypeError):
                raise ValueError(f"O índice de tempo t_index={self.t_index} não é válido.")
            if t_sample == 0:
                self.t_zero_indices.append(i)
            else:
                self.non_t_zero_indices.append(i)

        if not self.t_zero_indices or not self.non_t_zero_indices:
            raise ValueError("Não há amostras suficientes em cada classe para criar batches balanceados.")

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.t_zero_indices)
            random.shuffle(self.non_t_zero_indices)

        t_zero_iter = cycle(self.t_zero_indices)
        non_t_zero_iter = cycle(self.non_t_zero_indices)

        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            t_zero_batch = [next(t_zero_iter) for _ in range(2)] # Samples with t=0 per batch
            non_t_zero_batch = [next(non_t_zero_iter) for _ in range(self.batch_size-2)]
            batch_indices = t_zero_batch + non_t_zero_batch

            if self.shuffle:
                random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size

# %%
# Convertendo os dados para tensores
X_train = torch.as_tensor(df_train[features].values, dtype=torch.float32)
y_train = torch.as_tensor(df_train[target].values, dtype=torch.float32)

X_val = torch.as_tensor(df_val[features].values, dtype=torch.float32)
y_val = torch.as_tensor(df_val[target].values, dtype=torch.float32)

X_test = torch.as_tensor(df_test[features].values, dtype=torch.float32)
y_test = torch.as_tensor(df_test[target].values, dtype=torch.float32)

# Criando o Dataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Criando um loader para cada conjunto de dados
sampler = BalancedBatchSampler(train_dataset, batch_size=256)
train_loader = DataLoader(dataset=train_dataset,
                          batch_sampler=sampler,
                        #   shuffle=False,
                        #   batch_size=256,
                          )

val_loader = DataLoader(dataset=val_dataset,
                        shuffle=False,
                        batch_size=256
                        )

test_loader = DataLoader(dataset=test_dataset,
                        shuffle=False,
                        batch_size=len(test_dataset)
                        )

# %%
# Função de perda física ajustada
def physics_loss_fn(model, input_tensor, *args, **kwargs):
    # Separe as entradas e garanta que as variáveis requerem gradiente
    x = input_tensor[:, 0].view(-1, 1)  # x0
    z = input_tensor[:, 1].view(-1, 1)  # z0
    source_x0 = input_tensor[:, 2].view(-1, 1)  # theta0
    source_z0 = input_tensor[:, 3].view(-1, 1)  # t

    # Certifique-se de que todas as variáveis tenham requires_grad=True para calcular derivadas
    x.requires_grad_(True)
    z.requires_grad_(True)

    # Recalcule yhat usando as variáveis para garantir a conexão
    x_reconstructed = torch.cat([x, z, source_x0, source_z0], dim=1)
    yhat = model(x_reconstructed)
    
    travel_t = yhat[:, 0].view(-1, 1)

    # Calcule v(x, z)
    v = 1.5 + 0.3*x + 0.1*z
    
    # Calcule os gradientes necessários (dt/dx, dt/dz)
    dtdx = torch.autograd.grad(
        outputs=travel_t,  # Derivada de t em relação a x
        inputs=x,
        grad_outputs=torch.ones_like(x),
        create_graph=True,
        retain_graph=True
    )[0]

    dtdz = torch.autograd.grad(
        outputs=travel_t,  # Derivada de t em relação a z
        inputs=z,
        grad_outputs=torch.ones_like(z),
        create_graph=True,
        retain_graph=True
    )[0]


    ## (dt/dx)**2 + (dt/dz)**2 - (1/v)**2
    pde1 = dtdx**2 + dtdz**2 - (1/v)**2

    # Calcular a perda como a soma dos erros quadrados das PDEs
    loss = torch.mean(pde1 ** 2)

    return loss

# %%
torch.manual_seed(42)
input_size = len(features)
output_size = len(target)
model = KAN([input_size, 8, 6, 4, 2, output_size],
            grid_size=4,
            spline_order=3
            )

# Defines optimizer to update the parameters
optimizer = optim.Adam(model.parameters(),
                        lr=1e-3)

#Defines the Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=True
)

# Defines a MSE loss function
loss_fn = torch.nn.MSELoss(reduction='mean')

# %%
# arch = Architecture(model,
#                     loss_fn,
#                     physics_loss_fn,
#                     optimizer,
#                     lamb=0,
#                     lamb_l1=0,
#                     lamb_entropy=0,
#                     lambda_physics=1e2
#                     )
arch = Architecture(
    model=model,
    loss_fn=loss_fn,
    physics_fn=physics_loss_fn,
    optimizer=optimizer,
    scheduler=scheduler, 
    lamb=0,
    lamb_l1=0,
    lamb_entropy=0,
    lambda_physics=1e2
)

arch.set_loaders(train_loader, val_loader)

# %%
n_epochs = 50
arch.train(n_epochs=n_epochs, seed=SEED)

# %%
arch.save_checkpoint(f"output/checkpoint_{n_epochs}epochs.pt")

# %%
fig = arch.plot_losses()
fig.show()

# %%
predictions = arch.predict(df_train[features].values)
df_pred = pd.DataFrame(predictions, columns=target)
df_pred = df_test[features].join(df_pred)
df_pred.head()

# %%
df_test.loc[(df_test['source_x0'] == 1) & (df_test['source_z0'] == 2)]

# %%
score(df_pred[target].values, df_test[target].values, ["travel_time"])

# %% [markdown]
# ## Plotando os resultados

# %%
import seaborn as sns
def plot_heatmap_block(df, x_col, z_col, value_col, group_cols, block_index=0):
    """
    Plota um mapa de calor de um bloco específico do DataFrame agrupado.

    Parâmetros:
    - df: DataFrame do Pandas contendo os dados.
    - x_col: Nome da coluna para o eixo X.
    - z_col: Nome da coluna para o eixo Z.
    - value_col: Nome da coluna contendo os valores a serem plotados.
    - group_cols: Lista de colunas usadas para agrupar os blocos.
    - block_index: Índice do bloco a ser plotado (padrão: 0 - primeiro bloco).
    """
    # Agrupar os dados pelos pares especificados
    groups = list(df.groupby(group_cols))
    
    # Verificar se o índice está dentro do intervalo
    if block_index >= len(groups):
        print(f"Índice fora do intervalo. Escolha um número entre 0 e {len(groups) - 1}.")
        return
    
    # Obter o bloco solicitado
    (source_x0, source_z0), group = groups[block_index]
    
    # Plotar o mapa de calor
    plt.figure(figsize=(8, 6))
    pivot_table = group.pivot(index=z_col, columns=x_col, values=value_col)
    sns.heatmap(pivot_table, cmap='viridis')
    plt.title(f'Mapa de Calor: {group_cols[0]}={source_x0}, {group_cols[1]}={source_z0}')
    plt.show()

# %%
plot_heatmap_block(df, 'x', 'z', 'travel_time', ['source_x0', 'source_z0'], 1)

# %%
df_pykonal = df_test.loc[(df_test['source_x0'] == 1) & (df_test['source_z0'] == 2)]
df_pinnkan = df_pred.loc[(df_pred['source_x0'] == 1) & (df_pred['source_z0'] == 2)]

grid_pykonal = df_pykonal.pivot(index='z', columns='x', values='travel_time')
grid_pinnkan = df_pinnkan.pivot(index='z', columns='x', values='travel_time')

# %%
fig, (ax1, ax2) = plt.subplots(1, 2)  # 2 linhas, 1 coluna

ax1.imshow(grid_pykonal, cmap='terrain', interpolation='none')
ax1.set_title('Pykonal')

ax2.imshow(grid_pinnkan, cmap='terrain', interpolation='none')
ax2.set_title('PinnKan')

plt.tight_layout()  # Evita sobreposição de títulos
plt.show()

# %%
plt.imshow(df['travel_time'], cmap='terrain', interpolation='none')
plt.colorbar()  # Exibe a barra de cores para referência
plt.title("Heatmap da Matriz")
plt.show()


