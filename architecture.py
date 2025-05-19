import os
import random
import torch
from typing import Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from efficient_kan.kan import KAN
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as img
from matplotlib.figure import Figure


@dataclass
class MappedLoss:
    total_loss: float
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None
    reg_loss: Optional[float] = None

    def __init__(self, total_loss, data_loss=None, physics_loss=None, reg_loss=None):
        self.total_loss = total_loss.item()
        self.data_loss = data_loss.item() if data_loss else None
        self.physics_loss = physics_loss.item() if physics_loss else None
        self.reg_loss = reg_loss.item() if reg_loss else None


class Architecture(object):
    def __init__(self, model: KAN, loss_fn: callable, physics_fn: callable, optimizer, scheduler=None, use_weighted_pi=False, lamb=1e-2, lamb_l1=1., lamb_entropy=0., lambda_physics=1., device=None):
        super(Architecture, self).__init__()
        self.physics_loss_weight = lambda_physics
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.physics_fn = physics_fn
        self.optimizer = optimizer
        self.use_weighted_pi = use_weighted_pi
        self.clipping = None
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.losses = []
        self.p_losses = []
        self.d_losses = []
        self.reg_losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.scheduler = scheduler

        self.verbose = False
        self.warning_flag = False

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(
                f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_clip_backprop(self, clip_value):
        """Setup Value Clipping to Grandients"""
        if self.clipping is None:
            self.clipping = []
        for p in self.model.parameters():
            if p.requires_grad:
                handle = p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))
                self.clipping.append(handle)

    def remove_clip(self):
        """Remove Value Clipping to Grandients

        Use it after model training to avoid memory leaks
        """
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None

    @property
    def train_step_fn(self) -> callable:
        def perform_train_step_fn(x, y) -> MappedLoss:
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            input_features = self.model.layers[0].in_features
            if (x.shape[1] != input_features) and (self.warning_flag is False):
                self.warning_flag = True
                print(
                    "[WARNING] Input features are different from the model's input features. "
                    f"Only the first {input_features} features will be used as model input.")
            yhat = self.model(x[:, :input_features])
            yhat = torch.clamp(yhat, min=0)

            # Step 2 - Computes the data loss
            data_loss = self.loss_fn(yhat, y)

            # Step 3 - Computes the physics loss using x
            physics_loss = self.physics_fn(self.model, x)

            # Step 4 - Computes regularized loss
            reg_loss = self.model.regularization_loss(self.lamb_l1,
                                                      self.lamb_entropy)

            # print(f"{reg_loss=}")
            # print(f"{physics_loss=}")
            # print(f"{data_loss=}")

            # Step 5 - Sum all losses
            loss = data_loss + self.lamb * reg_loss
            if physics_loss is not None:
                loss += self.physics_loss_weight * physics_loss

            # Step 6 - Computes gradients for all parameters
            loss.backward()
            if callable(self.clipping):
                self.clipping()

            # Step 7 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Construct a torch.Tensor to return all losses
            # Ensure all values are on the same device as `loss`
            loss_tensor = torch.tensor(
                [
                    loss.item(),
                    data_loss.item(),
                    physics_loss.item() if physics_loss is not None else 0.0,
                    reg_loss.item(),
                ],
                device=self.device,
            )

            # Returns the loss
            return loss_tensor

        return perform_train_step_fn

    @property
    def val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)
            yhat = torch.clamp(yhat, min=0)

            # Step 2 - Computes the data loss
            data_loss = self.loss_fn(yhat, y)

            return data_loss

        return perform_val_step_fn

    def _mini_batch(self, validation=False) -> MappedLoss:
        # The mini-batch can be used with both loaders
        # The argument `validation` defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Compute the loss for the current batch
            loss_tensor = step_fn(x_batch, y_batch)

            # Append the loss tensor
            mini_batch_losses.append(loss_tensor)

        # Stack all mini-batch losses into a 2D tensor
        # Shape: (num_batches, len(loss_tensor))
        losses = torch.stack(mini_batch_losses)

        # Compute the mean along axis 0 (mean for each loss component)
        mean_losses = torch.mean(losses, dim=0)  # Shape: (4,)
        if mean_losses.dim() == 0:
            mean_losses = mean_losses.unsqueeze(0)

        if validation and self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(mean_losses[0].item())  # total_loss
            else:
                self.scheduler.step()

        return MappedLoss(*mean_losses)

    def _plot_kan_network(self, gamma=1, resolution=2, figsize=None,
                          folder="images") -> Figure:
        if figsize is None:
            figsize = (12, 24)
        num_points = 10**resolution
        num_layers = len(self.model.layers)
        layer_sizes = [layer.in_features for layer in self.model.layers]
        layer_sizes.append(self.model.layers[-1].out_features)

        # Adjusted the figure size for vertical layout
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        # Calculate neuron positions for vertical layout, centralized by the largest layer
        neuron_positions = {}
        max_neurons = max(layer_sizes)  # Get the size of the largest layer

        for layer_idx, layer_size in enumerate(layer_sizes):
            # Centralize neurons based on the largest layer
            x_positions = np.linspace(-max_neurons / 2,
                                      max_neurons / 2, layer_size)
            y = layer_idx  # Vertical position for each layer
            for neuron_idx in range(layer_size):
                if layer_sizes[layer_idx] == 1:
                    x_positions[neuron_idx] = 0
                neuron_positions[(layer_idx, neuron_idx)] = (
                    x_positions[neuron_idx], y)  # Assign neuron positions
                # Plot neurons as circles
                ax.plot(x_positions[neuron_idx], y, 'o',
                        color='black', markersize=8)

        # Dynamically set the zoom factor based on the distance between the neurons

        zoom_factor = figsize[0] / \
            (max_neurons * max_neurons * len(layer_sizes))

        # Plot connections and activation functions
        for layer_idx, layer in enumerate(self.model.layers):
            grid = layer.grid
            spline_weight = layer.scaled_spline_weight
            in_features = layer.in_features
            out_features = layer.out_features
            spline_order = layer.spline_order
            grid_size = layer.grid_size
            device = grid.device

            for in_feat in range(in_features):
                for out_feat in range(out_features):
                    # Start and end positions for the connection
                    start_pos = neuron_positions[(layer_idx, in_feat)]
                    end_pos = neuron_positions[(layer_idx + 1, out_feat)]

                    # Calculate the importance of the connection (e.g., norm of coefficients)
                    coeffs = spline_weight[out_feat, in_feat, :]
                    alpha = np.tanh(coeffs.abs().mean().item() * gamma)

                    # Plot connection line
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                            color='black', alpha=alpha, linewidth=2)

                    # Positioning the image (spline) at the midpoint of the connection
                    midpoint_x = (start_pos[0] + end_pos[0]) / 2
                    midpoint_y = (start_pos[1] + end_pos[1]) / 2

                    # Calculate the Euclidean distance between the two neurons
                    distance = np.sqrt(
                        (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)

                    # Add the spline image at the midpoint with the dynamic zoom
                    spline = img.imread(
                        f"{folder}/{layer_idx}_{in_feat}_{out_feat}.png")
                    # Set dynamic zoom factor
                    imagebox = OffsetImage(spline, zoom=zoom_factor)
                    ab = AnnotationBbox(
                        imagebox, (midpoint_x, midpoint_y), frameon=False)
                    ax.add_artist(ab)

        return fig

    def _save_kan_functions(self, folder="images", resolution=2):
        num_points = 10**resolution
        for i, layer in enumerate(self.model.layers):
            grid = layer.grid
            spline_weight = layer.scaled_spline_weight
            in_features = layer.in_features
            out_features = layer.out_features
            spline_order = layer.spline_order
            grid_size = layer.grid_size
            device = grid.device

            for in_feat in range(in_features):
                for out_feat in range(out_features):
                    coeffs = spline_weight[out_feat, in_feat, :]
                    grid_points = grid[in_feat, spline_order: -spline_order]
                    x_min = grid_points[0].item()
                    x_max = grid_points[-1].item()
                    x = torch.linspace(x_min, x_max, num_points, device=device)
                    y = torch.zeros_like(x)
                    x_input = x.unsqueeze(1).expand(-1, layer.in_features)
                    bases = layer.b_splines(x_input)[:, in_feat, :]
                    y = bases @ coeffs
                    x_cpu = x.cpu().numpy()
                    y_cpu = y.detach().cpu().numpy()

                    # Criar figura
                    plt.figure(figsize=(1, 1))
                    # Remover ticks
                    plt.xticks([])
                    plt.yticks([])

                    # Plotar o gráfico
                    plt.plot(x_cpu, y_cpu, color='black', linewidth=5)
                    plt.tight_layout()

                    # Salvar com o nome no formato
                    # "{id_camada}_{id_neuronio_entrada}_{id_neuronio_saida}.png"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    file_name = f"{folder}/{i}_{in_feat}_{out_feat}.png"
                    plt.savefig(file_name)
                    plt.close()

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self, n_epochs, seed=42, verbose=False):
        self.verbose = verbose

        # To ensure reproducibility of the training process
        self.set_seed(seed)

        total_epochs = self.total_epochs + n_epochs
        for epoch in (pbar := tqdm(range(self.total_epochs, total_epochs),
                                   total=total_epochs,
                                   initial=self.total_epochs,
                                   desc="Training PIKAN")):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # Performs training using mini-batches
            loss: MappedLoss = self._mini_batch(
                validation=False)
            self.losses.append(loss.total_loss)
            self.d_losses.append(loss.data_loss)
            self.p_losses.append(loss.physics_loss)
            self.reg_losses.append(loss.reg_loss)

            # Performs evaluation using mini-batches
            val_loss: MappedLoss = self._mini_batch(validation=True)
            self.val_losses.append(val_loss.total_loss)

            pbar.set_postfix(loss=loss.total_loss,
                             data_loss=loss.data_loss,
                             physics_loss=loss.physics_loss * self.physics_loss_weight,
                             val_loss=val_loss.total_loss)

    def save_checkpoint(self, filename, create_dirs=False):
        """
        Save the checkpoint to a specified file.

        Parameters:
            filename (str): Path to the file where the checkpoint will be saved.
            create_dirs (bool): Whether to create parent directories if they do not exist. Default is False.
        """
        # Create parent directories if the flag is set
        if create_dirs:
            dir_name = os.path.dirname(filename)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

        # Builds dictionary with all elements for resuming training
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'd_loss': self.d_losses,
            'p_loss': self.p_losses,
            'val_losses': self.val_losses
        }

        # Save the checkpoint
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename, map_location=self.device)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.d_losses = checkpoint['d_loss']
        self.p_losses = checkpoint['p_loss']
        self.val_losses = checkpoint['val_losses']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 4))
        axs[0].plot(self.losses, label='Training Loss', c='b')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(self.p_losses, label='Training Physics Loss', c='b')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        axs[2].plot(self.d_losses, label='Training Data Loss', c='b')
        axs[2].plot(self.val_losses, label='Validation Data Loss', c='r')
        axs[2].set_yscale('log')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        plt.tight_layout()
        return fig

    def plot_model(self, figsize=None, gamma=10, resolution=2, folder="images") -> Figure:
        """Plot KAN Architecture

        Args:
            figsize (tuple[int, int]): Figure size.
            gamma (int): Strength of line transparency. When too high, transparency
        is 1.
            resolution (int): Plot resolution.
            folder (str): Folder to save the b-spline plot as images.

        Returns:
            fig: Figure with the plot.

        Nodes are represented as circles.
        Line transparency is proportional to the importance of the connection.
        """
        self._save_kan_functions(folder=folder, resolution=resolution)
        fig = self._plot_kan_network(gamma=gamma, resolution=resolution,
                                     figsize=figsize, folder=folder)
        return fig
