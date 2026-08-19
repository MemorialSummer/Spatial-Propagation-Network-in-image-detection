import numpy as np
import matplotlib.pyplot as plt


def draw_layer(ax, x, n_draw, radius=0.08):
    ys = np.linspace(0.9, 0.1, n_draw)

    for y in ys:
        c = plt.Circle((x, y), radius,
                       fc="white",
                       ec="black",
                       lw=1.5)
        ax.add_patch(c)

    return ys
def connect(ax, x1, ys1, x2, ys2):

    for y1 in ys1:
        for y2 in ys2:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#B0B0B0",
                lw=0.5,
                zorder=0
            )
def draw_readout(ax):

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # 只画部分神经元，不然1000个输出神经元根本画不下
    n_input = 9
    n_hidden1 = 8
    n_hidden2 = 6
    n_output = 10

    x = [0.2, 1.3, 2.4, 3.5]

    ys0 = draw_layer(ax, x[0], n_input)
    ys1 = draw_layer(ax, x[1], n_hidden1)
    ys2 = draw_layer(ax, x[2], n_hidden2)
    ys3 = draw_layer(ax, x[3], n_output)

    connect(ax, x[0], ys0, x[1], ys1)
    connect(ax, x[1], ys1, x[2], ys2)
    connect(ax, x[2], ys2, x[3], ys3)

    # 标注
    ax.text(x[0], 1.05,
            "Output\nNeurons",
            ha="center",
            fontsize=12)

    ax.text(x[1], 1.05,
            "128\nReLU\nDropout",
            ha="center",
            fontsize=11)

    ax.text(x[2], 1.05,
            "64\nReLU\nDropout",
            ha="center",
            fontsize=11)

    ax.text(x[3], 1.05,
            "1000\nClasses",
            ha="center",
            fontsize=12)
