import numpy as np
import matplotlib.pyplot as plt
import torch

from typing import List


class PlotErrorSurfaces:
    
    def __init__(
        self, 
        w_range: float, 
        b_range: float, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        n_samples: int = 30, 
        plot_initial: bool = True
    ):
        self.x: np.ndarray = X.numpy()  # Converted to NumPy array for compatibility with NumPy operations
        self.y: np.ndarray = Y.numpy()
        self.w, self.b = np.meshgrid(
            np.linspace(-w_range, w_range, n_samples),
            np.linspace(-b_range, b_range, n_samples)
        )
        self.Z: np.ndarray = self.calculate_loss_surface()
        self.W: List[float] = []
        self.B: List[float] = []
        self.LOSS: List[float] = []
        self.iteration: int = 0

        if plot_initial:
            self.plot_surface()
            self.plot_contour()

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def calculate_loss_surface(self) -> np.ndarray:
        return np.array([
            [np.mean((self.y - self.sigmoid(w * self.x + b)) ** 2) for w, b in zip(w_row, b_row)]
            for w_row, b_row in zip(self.w, self.b)
        ])

    def update_params(self, model: torch.nn.Module, loss: float) -> None:
        self.iteration += 1
        self.W.append(model.linear.weight.item())
        self.B.append(model.linear.bias.item())
        self.LOSS.append(loss)

    def plot_surface(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.w, self.b, self.Z, cmap='viridis')
        ax.set_title('Loss Surface')
        ax.set_xlabel('Weight (w)')
        ax.set_ylabel('Bias (b)')
        plt.show()

    def plot_contour(self) -> None:
        plt.contour(self.w, self.b, self.Z)
        plt.xlabel('Weight (w)')
        plt.ylabel('Bias (b)')
        plt.title('Loss Surface Contour')
        plt.show()

    def final_plot(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z, color='gray')
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x')
        plt.show()

        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('Weight (w)')
        plt.ylabel('Bias (b)')
        plt.title('Training Path on Loss Surface')
        plt.show()

# Auxiliary plotting function
def plot_progress(X: torch.Tensor, Y: torch.Tensor, model: torch.nn.Module, epoch: int) -> None:
    plt.plot(X.numpy(), model(X).detach().numpy(), label=f'Epoch {epoch}')
    plt.plot(X.numpy(), Y.numpy(), 'r', label='True Data')
    plt.legend()
    plt.show()
