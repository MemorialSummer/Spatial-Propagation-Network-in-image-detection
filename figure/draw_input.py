"""
draw_input.py

Draw the left input encoding module of the SPN architecture.

Author: ChatGPT
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe


# ============================================================
# Colors
# ============================================================

R_COLOR = "#E74C3C"
G_COLOR = "#2ECC71"
B_COLOR = "#3498DB"

TEXT_COLOR = "#1F2937"
PANEL_COLOR = "#F8FAFC"
SHADOW_COLOR = "#D1D5DB"
EDGE_COLOR = "#4B5563"
PIXEL_COLOR = "#FDE68A"
NEURON_RED = "#E53935"
LINE_COLOR = "#64748B"


# ============================================================
# Parameters
# ============================================================

DRAW_GRID = 16
INPUT_PLANE = 10
STACK_OFFSET = 0.35
GRID_SIZE = 4.8
PLANE_SIZE = 3.6


# ============================================================
# Helpers
# ============================================================

def add_text(ax, x, y, text, size=12, color=TEXT_COLOR, weight="normal", alpha=1.0):
    txt = ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        alpha=alpha,
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=4, foreground="white", alpha=0.85),
    ])
    return txt


# ============================================================
# Draw One RGB Grid
# ============================================================

def draw_single_grid(ax, x, y, color, label):
    rect = FancyBboxPatch(
        (x, y),
        GRID_SIZE,
        GRID_SIZE,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="white",
        edgecolor=color,
        linewidth=2.2,
        zorder=2,
    )
    shadow = FancyBboxPatch(
        (x + 0.06, y - 0.06),
        GRID_SIZE,
        GRID_SIZE,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=SHADOW_COLOR,
        edgecolor="none",
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(shadow)
    ax.add_patch(rect)

    step = GRID_SIZE / DRAW_GRID
    for i in range(1, DRAW_GRID):
        ax.plot(
            [x + i * step, x + i * step],
            [y, y + GRID_SIZE],
            color=color,
            lw=0.35,
            alpha=0.45,
            zorder=2,
        )
        ax.plot(
            [x, x + GRID_SIZE],
            [y + i * step, y + i * step],
            color=color,
            lw=0.35,
            alpha=0.45,
            zorder=2,
        )

    add_text(ax, x + GRID_SIZE + 0.25, y + GRID_SIZE / 2, label, size=16, color=color, weight="bold")


# ============================================================
# Highlight Example Region
# ============================================================

def highlight_region(ax, x, y):
    cell = GRID_SIZE / DRAW_GRID
    region = Rectangle(
        (x, y + GRID_SIZE - 3 * cell),
        3 * cell,
        3 * cell,
        facecolor=PIXEL_COLOR,
        edgecolor="#F59E0B",
        linewidth=1.8,
        alpha=0.75,
        zorder=3,
    )
    ax.add_patch(region)

    glow = Rectangle(
        (x - 0.06, y + GRID_SIZE - 3 * cell - 0.06),
        3 * cell + 0.12,
        3 * cell + 0.12,
        facecolor="#FDE68A",
        edgecolor="none",
        alpha=0.2,
        zorder=2,
    )
    ax.add_patch(glow)


# ============================================================
# Draw RGB Stack
# ============================================================

def draw_rgb_stack(ax):
    bx = 0.8
    by = 1.1

    draw_single_grid(ax, bx, by, B_COLOR, "B")
    draw_single_grid(ax, bx + STACK_OFFSET, by + STACK_OFFSET, G_COLOR, "G")
    draw_single_grid(ax, bx + STACK_OFFSET * 2, by + STACK_OFFSET * 2, R_COLOR, "R")
    highlight_region(ax, bx + STACK_OFFSET * 2, by + STACK_OFFSET * 2)

    add_text(ax, bx + 1.6, by + GRID_SIZE + 1.0, "224×224 RGB Image", size=15, weight="bold")


# ============================================================
# Draw Enlarged Region
# ============================================================

def draw_region_example(ax):
    ox = 7.3
    oy = 3.0
    spacing = 0.8


    for i in range(3):
        for j in range(3):
            xx = ox + j * spacing
            yy = oy + (2 - i) * spacing

            ax.add_patch(Circle((xx, yy), 0.06, color=R_COLOR, zorder=5))

            if j < 2:
                ax.plot([xx, xx + spacing], [yy, yy], color=R_COLOR, lw=1.0, zorder=4)
            if i < 2:
                ax.plot([xx, xx], [yy, yy - spacing], color=R_COLOR, lw=1.0, zorder=4)

            add_text(ax, xx - 0.14, yy + 0.12, f"({i},{j})", size=6.5, color="#7F1D1D")

    add_text(ax, ox + 0.5, oy - 0.9, "Average", size=12.5, weight="bold")

    start_x = ox + 1.0
    start_y = oy - 0.45
    end_x = 13.15
    end_y = 3.7

    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color=LINE_COLOR,
        connectionstyle="arc3,rad=-0.08",
        zorder=6,
    )
    ax.add_patch(arrow)


# ============================================================
# Draw 10×10 Red Input Neuron Plane
# ============================================================

def draw_input_plane(ax):
    plane_x = 13.4
    plane_y = 1.0
    step = PLANE_SIZE / (INPUT_PLANE - 1)

    panel = FancyBboxPatch(
        (plane_x - 0.35, plane_y - 0.35),
        PLANE_SIZE + 0.7,
        PLANE_SIZE + 0.7,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(panel)

    add_text(ax, plane_x - 0.25, plane_y + PLANE_SIZE + 0.8, "Input Neuron Plane", size=14, weight="bold")

    neuron_pos = []
    for i in range(INPUT_PLANE):
        row = []
        for j in range(INPUT_PLANE):
            x = plane_x + j * step
            y = plane_y + (INPUT_PLANE - 1 - i) * step
            row.append((x, y))

            halo = Circle((x, y), radius=0.085, facecolor="#FDE2E2", edgecolor="none", alpha=0.65, zorder=3)
            ax.add_patch(halo)
            c = Circle((x, y), radius=0.055, facecolor=NEURON_RED, edgecolor="#7F1D1D", linewidth=0.3, zorder=5)
            ax.add_patch(c)
        neuron_pos.append(row)

    return neuron_pos


# ============================================================
# Draw Connection Example
# ============================================================

def draw_region_to_neuron(ax, neuron_pos):
    region_x = 7.3
    region_y = 3.0
    spacing = 0.8
    points = []

    for i in range(3):
        for j in range(3):
            xx = region_x + j * spacing
            yy = region_y + (2 - i) * spacing
            points.append((xx, yy))

    neuron = neuron_pos[0][0]
    for px, py in points:
        line = FancyArrowPatch(
            (px, py),
            neuron,
            arrowstyle="-",
            linewidth=0.9,
            color=R_COLOR,
            alpha=0.35,
            connectionstyle="arc3,rad=0.08",
            zorder=4,
        )
        ax.add_patch(line)

    ax.add_patch(Circle(neuron, radius=0.11, fill=False, edgecolor="#F59E0B", linewidth=2.5, zorder=8))


# ============================================================
# Labels
# ============================================================

def draw_labels(ax):
    add_text(ax, 17.8, 4.05, "10×10", size=11.5, weight="bold")
    add_text(ax, 17.8, 3.45, "Input Plane", size=10.5)
    add_text(ax, 17.8, 2.85, "R Channel", size=10.5, color=R_COLOR, weight="bold")
    add_text(ax, 1.0, 7.95, "Spatial Encoding", size=14, weight="bold")


# ============================================================
# Main API
# ============================================================

def draw_input(ax):
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.2)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_COLOR)
    ax.axis("off")

    draw_rgb_stack(ax)
    draw_region_example(ax)
    neurons = draw_input_plane(ax)
    draw_region_to_neuron(ax, neurons)
    draw_labels(ax)
