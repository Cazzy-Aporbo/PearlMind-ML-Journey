
# src/visual_theme_demo.py
# Demonstrates color usage and required call:
#   xplot_colortable(mcolors.CSS4_COLORS); plt.show()
# and example plots using greys, pinks, purples, teals, plus a blue ombré heatmap.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def xplot_colortable(colors, ncols=4, sort_colors=True):
    if sort_colors:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors.keys())

    n = len(names)
    nrows = n // ncols + int(n % ncols > 0)
    cell_w, cell_h = 18, 0.35
    swatch_w, swatch_h = 18, 0.25
    fig, ax = plt.subplots(figsize=(cell_w, nrows*cell_h))
    ax.set_xlim(0, cell_w)
    ax.set_ylim(0, nrows*cell_h)
    ax.axis('off')

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = nrows - row - 1
        xi_text = col * (cell_w / ncols) + swatch_w + 0.5
        xi_swatch = col * (cell_w / ncols)
        color = colors[name]
        ax.text(xi_text, y*cell_h + swatch_h/2, name, fontsize=8, ha='left', va='center')
        ax.add_patch(plt.Rectangle((xi_swatch, y*cell_h), swatch_w, swatch_h, facecolor=color, edgecolor='0.8'))
    fig.tight_layout()
    return fig, ax

# 1) Show full CSS4 colortable (required call)
fig, ax = xplot_colortable(mcolors.CSS4_COLORS)
plt.show()

# 2) Define constrained palettes
greys  = ["#111111", "#444444", "#777777", "#AAAAAA", "#DDDDDD"]
pinks  = ["#6B0F1A", "#B91372", "#F25287", "#FF9EB5", "#FFD1DC"]
purples= ["#2E004F", "#5E239D", "#7A3E9D", "#A076F9", "#DCC6FF"]
teals  = ["#003B46", "#07575B", "#66A5AD", "#C4DFE6", "#E0FBFC"]

# 3) Example line plot using greys
x = np.linspace(0, 10, 200)
plt.figure()
for i, c in enumerate(greys):
    plt.plot(x, np.sin(x + i*0.3) + i*0.2, linewidth=2, color=c, label=f"grey {i+1}")
plt.title("Greys Line Family")
plt.legend()
plt.show()

# 4) Example scatter using pinks
rng = np.random.default_rng(42)
x = rng.normal(0, 1, 200)
y = rng.normal(0, 1, 200)
plt.figure()
for i, c in enumerate(pinks):
    sel = slice(i*40, (i+1)*40)
    plt.scatter(x[sel], y[sel], s=40+10*i, color=c, alpha=0.8, label=f"pink {i+1}")
plt.title("Pinks Scatter Clusters")
plt.legend()
plt.show()

# 5) Example bar with purples
vals = np.abs(rng.normal(1.0, 0.3, size=5))
plt.figure()
plt.bar(range(5), vals, color=purples, edgecolor="#222222")
plt.title("Purples Bar Chart")
plt.show()

# 6) Example histogram with teals
data = rng.normal(0, 1, 1000)
plt.figure()
plt.hist(data, bins=20, color=teals[-2], edgecolor=teals[0], alpha=0.9)
plt.title("Teal Histogram")
plt.show()

# 7) Blue ombré gradient heatmap
blue_ombre = LinearSegmentedColormap.from_list("blue_ombre", ["#001F3F", "#005B96", "#0074D9", "#7FDBFF", "#E0F7FF"])
grid = np.outer(np.linspace(0,1,100), np.ones(100))
plt.figure()
plt.imshow(grid, aspect='auto', cmap=blue_ombre, origin='lower')
plt.title("Blue Ombré Gradient")
plt.axis('off')
plt.show()
