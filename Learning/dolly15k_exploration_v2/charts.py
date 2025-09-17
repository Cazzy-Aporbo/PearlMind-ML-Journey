
import matplotlib.pyplot as plt
from theme import PALETTE, TITLE_COLOR, GRID_ALPHA, FIGSIZE, FONT_SIZE

def bar_counts(counts, title, outpath, color_key="lavender", edge_key="lavender_edge"):
    labels, values = zip(*counts.items()) if counts else ([], [])
    plt.figure(figsize=FIGSIZE)
    if labels:
        plt.bar(labels, values, color=PALETTE[color_key], edgecolor=PALETTE.get(edge_key, "#222222"))
    plt.title(title, fontsize=14, color=TITLE_COLOR)
    plt.xticks(rotation=30, ha='right', fontsize=FONT_SIZE)
    plt.ylabel("Count", fontsize=FONT_SIZE)
    plt.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def bar_means(means, title, outpath, color_key="teal", edge_key="teal_edge", ylabel="Mean"):
    labels, values = zip(*means.items()) if means else ([], [])
    plt.figure(figsize=FIGSIZE)
    if labels:
        plt.bar(labels, values, color=PALETTE[color_key], edgecolor=PALETTE.get(edge_key, "#222222"))
    plt.title(title, fontsize=14, color=TITLE_COLOR)
    plt.xticks(rotation=30, ha='right', fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)
    plt.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def hist_series(series, title, outpath, color="#FFD6E8", edge="#6B5B95", bins=50, xlabel="Length (chars)"):
    plt.figure(figsize=FIGSIZE)
    try:
        plt.hist(series, bins=bins, color=color, edgecolor=edge)
    except Exception:
        pass
    plt.title(title, fontsize=14, color=TITLE_COLOR)
    plt.xlabel(xlabel, fontsize=FONT_SIZE)
    plt.ylabel("Count", fontsize=FONT_SIZE)
    plt.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
