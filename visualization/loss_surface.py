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
        self.x: np.ndarray = X.numpy()
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
            self.plot_side_by_side()

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

    def plot_side_by_side(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 3D Surface Plot
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax3d.plot_surface(self.w, self.b, self.Z, cmap='viridis')
        ax3d.set_title('Loss Surface')
        ax3d.set_xlabel('Weight (w)')
        ax3d.set_ylabel('Bias (b)')
        
        # 2D Contour Plot
        axes[1].contour(self.w, self.b, self.Z)
        axes[1].set_title('Loss Surface Contour')
        axes[1].set_xlabel('Weight (w)')
        axes[1].set_ylabel('Bias (b)')

        # Remove the outer frame (spines) of the 2D contour plot as well
        axes[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in axes[0].spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_update(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.plot(self.x, self.sigmoid(self.W[-1] * self.x + self.B[-1]), label="sigmoid")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.legend()
        plt.title(f'Data space iteration: {self.iteration}')
        
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title(f'Loss surface contour iteration {self.iteration}')
        plt.xlabel('w')
        plt.ylabel('b')

        plt.show()

    def final_plot(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 3D Wireframe Plot with Training Path
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax3d.plot_wireframe(self.w, self.b, self.Z, color='gray')
        ax3d.plot_surface(self.w, self.b, self.Z, cmap='viridis', alpha=0.5)
        ax3d.scatter(self.W, self.B, self.LOSS, c='r', marker='x')
        ax3d.set_title('Training Path on Loss Surface')
        ax3d.set_xlabel('Weight (w)')
        ax3d.set_ylabel('Bias (b)')

        # 2D Contour Plot with Training Path
        axes[1].contour(self.w, self.b, self.Z)
        axes[1].scatter(self.W, self.B, c='r', marker='x')
        axes[1].set_title('Training Path on Contour')
        axes[1].set_xlabel('Weight (w)')
        axes[1].set_ylabel('Bias (b)')

        # Remove the outer frame (spines) of the 2D contour plot as well
        axes[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in axes[0].spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.show()