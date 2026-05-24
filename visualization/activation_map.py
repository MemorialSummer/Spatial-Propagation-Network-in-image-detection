import torch
import numpy as np
import matplotlib.pyplot as plt

from models.spatial_network import SpatialNetwork
from configs.config import *


def visualize_activation(h):

    h = h.detach().cpu().numpy()

    h = h.reshape(
        GRID_X,
        GRID_Y,
        GRID_Z
    )

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = np.where(np.abs(h) > 0.2)

    values = h[xs, ys, zs]

    scatter = ax.scatter(
        xs,
        ys,
        zs,
        c=values,
        cmap='hot'
    )

    plt.colorbar(scatter)

    plt.show()