import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

from config import GRID_X, GRID_Y, GRID_Z, LOCAL_RADIUS, LONG_RANGE_CONNECTIONS


# ============================================================
# Style
# ============================================================

CUBE_COLOR = "#CBD5E1"
NEURON_COLOR = "#9CA3AF"
LOCAL_COLOR = "#2563EB"
SAME_LAYER_COLOR = "#10B981"
LONG_RANGE_COLOR = "#F59E0B"
OUTPUT_COLOR = "#8B5CF6"
INPUT_COLORS = ["#E74C3C", "#2ECC71", "#3498DB"]
TEXT_COLOR = "#1F2937"


# ============================================================
# 3D -> 2D projection
# ============================================================

def project_point(x, y, z, origin=(3.5, 0.2), scale=0.55, angle=20):
    theta = np.deg2rad(angle)

    # 绕 y 轴旋转
    xr = x * np.cos(theta) - z * np.sin(theta)
    zr = x * np.sin(theta) + z * np.cos(theta)

    # 正交投影
    u = origin[0] + scale * (zr - xr)
    v = origin[1] + scale * (y + 0.5 * (xr + zr))

    return u, v
# def project_point(x, y, z, origin=(4.25, 0.8), scale=0.55):
#     u = origin[0] + scale * (1.35 * z - 0.65 * x)
#     v = origin[1] + scale * (y + 0.35 * x + 0.65 * z)
#     return u, v

# ============================================================
# Helpers
# ============================================================

def draw_edge(ax, p1, p2, color, lw=1.2, alpha=0.95, zorder=2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, alpha=alpha, zorder=zorder)


def draw_wireframe_cube(ax, x0, y0, z0, size, color, lw=1.2, alpha=0.95):
    corners = [
        (x0, y0, z0),
        (x0 + size, y0, z0),
        (x0, y0 + size, z0),
        (x0, y0, z0 + size),
        (x0 + size, y0 + size, z0),
        (x0 + size, y0, z0 + size),
        (x0, y0 + size, z0 + size),
        (x0 + size, y0 + size, z0 + size),
    ]
    points = {i: project_point(*corner) for i, corner in enumerate(corners)}
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    for a, b in edges:
        draw_edge(ax, points[a], points[b], color=color, lw=lw, alpha=alpha, zorder=2)


# ============================================================
# 1) Draw main cube
# ============================================================

def draw_cube(ax):
    draw_wireframe_cube(ax, 0, 0, 0, 9, color=CUBE_COLOR, lw=1.2, alpha=0.95)


def draw_background_grid(ax):
    grid_points = list(range(10))
    for x in grid_points:
        for z in grid_points:
            p1 = project_point(x, 0, z)
            p2 = project_point(x, 9, z)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#94A3B8", lw=0.4, alpha=0.12, zorder=0)
        for y in grid_points:
            p1 = project_point(x, y, 0)
            p2 = project_point(x, y, 9)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#94A3B8", lw=0.4, alpha=0.12, zorder=0)
    for y in grid_points:
        for z in grid_points:
            p1 = project_point(0, y, z)
            p2 = project_point(9, y, z)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#94A3B8", lw=0.4, alpha=0.12, zorder=0)


# ============================================================
# 2) Draw neurons (subsampled)
# ============================================================

def draw_neurons(ax):
    for x in range(10):
        for y in range(10):
            for z in range(10):
                u, v = project_point(x, y, z)
                if z == 0:
                    color = INPUT_COLORS[0]
                    alpha = 0.95
                elif z == 1:
                    color = INPUT_COLORS[1]
                    alpha = 0.95
                elif z == 2:
                    color = INPUT_COLORS[2]
                    alpha = 0.95
                else:
                    color = NEURON_COLOR
                    alpha = 0.6
                ax.scatter([u], [v], s=8, color=color, alpha=alpha, zorder=5)


# ============================================================
# 3) Left input plane (R/G/B)
# ============================================================

def draw_input_plane(ax):
    # draw full xy planes for z=0, z=1, z=2
    for z in range(3):
        plane = [
            project_point(0, 0, z),
            project_point(9, 0, z),
            project_point(9, 9, z),
            project_point(0, 9, z),
        ]
        ax.add_patch(Polygon(plane, closed=True, facecolor=INPUT_COLORS[z], edgecolor="none", alpha=0.18, zorder=1))

    # suggest 10x10 grid on the input planes
    for z in range(3):
        for x in range(10):
            p1 = project_point(x, 0, z)
            p2 = project_point(x, 9, z)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#D1D5DB", lw=0.4, alpha=0.6, zorder=1)
        for y in range(10):
            p1 = project_point(0, y, z)
            p2 = project_point(9, y, z)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#D1D5DB", lw=0.4, alpha=0.6, zorder=1)

    red_label = project_point(5, 10.5, 0)
    green_label = project_point(5, 10.5, 1)
    blue_label = project_point(5, 10.5, 2)
    ax.text(red_label[0] - 0.8, red_label[1] + 0.05, "Red input", fontsize=9, color=INPUT_COLORS[0], weight="bold")
    ax.text(green_label[0] - 0.8, green_label[1] + 0.05, "Green input", fontsize=9, color=INPUT_COLORS[1], weight="bold")
    ax.text(blue_label[0] - 0.8, blue_label[1] + 0.05, "Blue input", fontsize=9, color=INPUT_COLORS[2], weight="bold")


# ============================================================
# 4) Representative connectivity drawing
# ============================================================

REPRESENTATIVE_NEURON = (3, 4, 0)
LONG_TARGETS = [(0, 0, 9), (9, 9, 0), (9, 0, 9)]


def get_layer(x, y, z, grid_x, grid_y, grid_z):
    return min(x, y, z, grid_x - 1 - x, grid_y - 1 - y, grid_z - 1 - z)


def get_neighbors_within_radius(x, y, z, grid_x, grid_y, grid_z):
    neighbors = []
    for dx in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
        for dy in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            for dz in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx = x + dx
                ny = y + dy
                nz = z + dz
                if 0 <= nx < grid_x and 0 <= ny < grid_y and 0 <= nz < grid_z:
                    neighbors.append((nx, ny, nz))
    return neighbors


def draw_arrow(ax, src, tgt, color, lw=1.8, alpha=0.98, zorder=4):
    arrow = FancyArrowPatch(
        src,
        tgt,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def draw_edge_group(ax, edges, base_color, highlight_edge=None, highlight_color=None, lw=1.0, alpha=0.42, highlight_lw=2.0, highlight_alpha=0.95, zorder=3):
    for src, tgt in edges:
        p1 = project_point(*src)
        p2 = project_point(*tgt)
        draw_arrow(ax, p1, p2, color=base_color, lw=lw, alpha=alpha, zorder=zorder)

    if highlight_edge is not None:
        p1 = project_point(*highlight_edge[0])
        p2 = project_point(*highlight_edge[1])
        draw_arrow(ax, p1, p2, color=highlight_color, lw=highlight_lw, alpha=highlight_alpha, zorder=zorder + 1)


def draw_local_connections(ax):
    neighbors = get_neighbors_within_radius(*REPRESENTATIVE_NEURON, GRID_X, GRID_Y, GRID_Z)
    src_layer = get_layer(*REPRESENTATIVE_NEURON, GRID_X, GRID_Y, GRID_Z)
    inner_edges = [(REPRESENTATIVE_NEURON, (nx, ny, nz)) for nx, ny, nz in neighbors if get_layer(nx, ny, nz, GRID_X, GRID_Y, GRID_Z) > src_layer]
    highlight_edge = inner_edges[0] if inner_edges else None
    draw_edge_group(ax, inner_edges, "#9CA3AF", highlight_edge=highlight_edge, highlight_color=LOCAL_COLOR)


def draw_same_layer(ax):
    z = REPRESENTATIVE_NEURON[2]
    x0, y0 = 2, 3
    edges = [((x, y0, z), (x + 1, y0, z)) for x in range(x0, x0 + 4)]
    edges += [((x0, y, z), (x0, y + 1, z)) for y in range(y0 - 1, y0 + 3)]
    highlight_edge = edges[0] if edges else None
    draw_edge_group(ax, edges, "#9CA3AF", highlight_edge=highlight_edge, highlight_color=SAME_LAYER_COLOR)


def draw_long_range(ax):
    edges = [(REPRESENTATIVE_NEURON, tgt) for tgt in LONG_TARGETS]
    highlight_edge = edges[0]
    draw_edge_group(ax, edges, "#9CA3AF", highlight_edge=highlight_edge, highlight_color=LONG_RANGE_COLOR)


# ============================================================
# 7) Output neuron region
# ============================================================

def draw_output_cube(ax):
    draw_wireframe_cube(ax, 3.5, 3.5, 3.5, 3.0, color=OUTPUT_COLOR, lw=1.4, alpha=0.85)

    for x in [4.0, 5.0, 6.0]:
        for y in [4.0, 5.0, 6.0]:
            for z in [4.0, 5.0, 6.0]:
                u, v = project_point(x, y, z)
                ax.scatter([u], [v], s=18, color=OUTPUT_COLOR, zorder=6)


# ============================================================
# Annotations
# ============================================================

def draw_annotations(ax):
    ax.text(8.8, 7.2, "Representative local connections", fontsize=10.2, color=LOCAL_COLOR, ha="left")
    ax.text(8.8, 5.1, "Representative same-layer propagation", fontsize=10.2, color=SAME_LAYER_COLOR, ha="left")
    ax.text(8.8, 3.0, "Representative long-range connections", fontsize=10.2, color=LONG_RANGE_COLOR, ha="left")
    ax.text(8.8, 1.0, "Output\nNeuron Region", fontsize=10.2, color=OUTPUT_COLOR, ha="left")


# ============================================================
# Main API
# ============================================================

def project_point_333(x, y, z, origin=(1.8, 1.1), scale=1.15, angle=20):
    theta = np.deg2rad(angle)
    xr = x * np.cos(theta) - z * np.sin(theta)
    zr = x * np.sin(theta) + z * np.cos(theta)
    u = origin[0] + scale * (zr - xr)
    v = origin[1] + scale * (y + 0.45 * (xr + zr))
    return u, v


def draw_wireframe_cube_333(ax, x0, y0, z0, size, color, lw=1.0, alpha=0.75):
    nodes = [(x, y, z) for x in range(3)
                     for y in range(3)
                     for z in range(3)]

    pts = [project_point_333(x, y, z) for x, y, z in nodes]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    margin = 0.3

    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)
    corners = [
        (x0, y0, z0),
        (x0 + size, y0, z0),
        (x0, y0 + size, z0),
        (x0, y0, z0 + size),
        (x0 + size, y0 + size, z0),
        (x0 + size, y0, z0 + size),
        (x0, y0 + size, z0 + size),
        (x0 + size, y0 + size, z0 + size),
    ]
    points = {i: project_point_333(*corner) for i, corner in enumerate(corners)}
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    for a, b in edges:
        ax.plot(
            [points[a][0], points[b][0]],
            [points[a][1], points[b][1]],
            color=color,
            lw=lw,
            alpha=alpha,
            zorder=1,
        )


def draw_full_3x3x3_example(ax):
    ax.set_xlim(-0.1, 3.6)
    ax.set_ylim(-0.1, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    draw_wireframe_cube_333(ax, 0.0, 0.0, 0.0, 2.0, color=CUBE_COLOR, lw=0.9, alpha=0.7)

    nodes = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    for x, y, z in nodes:
        u, v = project_point_333(x, y, z)
        ax.scatter([u], [v], s=16, color="#6B7280", alpha=1.0, zorder=5)

    for src in nodes:
        x0, y0, z0 = src
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == dy == dz == 0:
                        continue
                    tgt = (x0 + dx, y0 + dy, z0 + dz)
                    if tgt not in nodes:
                        continue
                    p1 = project_point_333(*src)
                    p2 = project_point_333(*tgt)
                    arrow = FancyArrowPatch(
                        p1,
                        p2,
                        arrowstyle="-|>",
                        mutation_scale=7,
                        lw=0.45,
                        color="#4B5563",
                        alpha=0.65,
                        zorder=2,
                        connectionstyle="arc3,rad=0.0",
                    )
                    ax.add_patch(arrow)

    ax.text(1.8, 3.75, "3×3×3 local neighborhood (r=1)", fontsize=9, color=TEXT_COLOR, ha="center")


def draw_spn(ax, show_full_example=False):
    if show_full_example:
        draw_full_3x3x3_example(ax)
        return

    ax.set_xlim(-1.5, 10.0)
    ax.set_ylim(-1.0, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    draw_background_grid(ax)
    draw_cube(ax)
    draw_input_plane(ax)
    draw_neurons(ax)
    draw_local_connections(ax)
    draw_same_layer(ax)
    draw_long_range(ax)
    draw_output_cube(ax)
    draw_annotations(ax)
