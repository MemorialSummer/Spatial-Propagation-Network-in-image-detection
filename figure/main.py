import matplotlib.pyplot as plt

from draw_input import draw_input
from draw_spn import draw_spn
from draw_readout import draw_readout


def main():

    # ============================================================
    # Global Figure Style
    # ============================================================

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    # 创建画布（论文常用横向比例）
    fig = plt.figure(figsize=(20, 10), facecolor="white")

    # ============================================================
    # Layout
    # ============================================================

    ax_input = fig.add_axes([0.03, 0.52, 0.43, 0.42])
    ax_spn = fig.add_axes([0.50, 0.52, 0.46, 0.42])
    ax_example = fig.add_axes([0.03, 0.05, 0.43, 0.42])
    ax_readout = fig.add_axes([0.50, 0.05, 0.46, 0.42])

    # ============================================================
    # Draw Modules
    # ============================================================

    draw_input(ax_input)
    draw_spn(ax_spn)
    draw_spn(ax_example, show_full_example=True)
    draw_readout(ax_readout)

    # ============================================================
    # Titles
    # ============================================================

    fig.text(0.245, 0.95, "1. Input Encoding",
         ha="center",
         fontsize=16,
         weight="bold")

    fig.text(0.73, 0.95, "2. Spatial Propagation Network",
            ha="center",
            fontsize=16,
            weight="bold")

    fig.text(0.245, 0.48, "3. 3×3×3 Fully Connected Example",
            ha="center",
            fontsize=16,
            weight="bold")

    fig.text(0.73, 0.48, "4. Readout",
            ha="center",
            fontsize=16,
            weight="bold")

    # ============================================================
    # Hide axes
    # ============================================================

    ax_input.set_axis_off()

    # ============================================================
    # Save
    # ============================================================

    plt.savefig(
        "SPN_Architecture.pdf",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "SPN_Architecture.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()