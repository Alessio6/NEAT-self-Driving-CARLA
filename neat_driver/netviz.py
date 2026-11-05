"""Matplotlib window that draws a NEAT network and the signals flowing through it.

Inputs are on the left, hidden nodes in the middle (by layer), outputs on the
right. Connections are green for positive weights and red for negative ones,
thicker for larger weights. With live signals, nodes and connections are
coloured by their current value (blue = negative, red = positive).
"""

import matplotlib
import matplotlib.ticker
import numpy as np
from neat.graphs import feed_forward_layers

from .env import INPUT_LABELS, OUTPUT_LABELS

POSITIVE, NEGATIVE = "#2e9e5b", "#d6453d"
INPUT_COLOUR, HIDDEN_COLOUR, OUTPUT_COLOUR, UNUSED_COLOUR = "#9db4d8", "#b58ad6", "#f2b84b", "#dddddd"


class NetworkView:
    def __init__(self, title="NEAT network", interactive=True, show_history=False):
        """`interactive=False` draws off-screen, for saving pictures only."""
        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # after choosing the backend
        from matplotlib.collections import LineCollection

        self.plt, self.LineCollection = plt, LineCollection
        self.interactive = interactive
        self.cmap = plt.get_cmap("coolwarm")
        if interactive:
            plt.ion()
        if show_history:
            self.fig, (self.ax, self.history_ax) = plt.subplots(
                2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [4, 1]})
        else:
            self.fig, self.ax = plt.subplots(figsize=(9, 7))
            self.history_ax = None
        self.fig.canvas.manager.set_window_title(title)
        self.node_keys, self.edges = [], []
        if interactive:
            self.fig.show()

    # ---- Window ---------------------------------------------------------------
    def is_open(self):
        return not self.interactive or self.plt.fignum_exists(self.fig.number)

    def refresh(self):
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=110)

    # ---- Structure ------------------------------------------------------------
    def show_genome(self, genome, config, info="", history=None):
        """Redraw the network for `genome`; call once per genome."""
        inputs, outputs = config.genome_config.input_keys, config.genome_config.output_keys
        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(-0.32, 1.42)
        ax.set_ylim(-0.06, 1.1)

        positions, unused = _layout(genome, inputs, outputs)
        self.edges = [(c.key, c.weight) for c in genome.connections.values()
                      if c.enabled and c.key[0] in positions and c.key[1] in positions]
        weights = np.array([w for _, w in self.edges])
        self.edge_lines = self.LineCollection(
            [(positions[i], positions[o]) for (i, o), _ in self.edges],
            colors=[POSITIVE if w > 0 else NEGATIVE for w in weights],
            linewidths=0.4 + 2.6 * np.clip(np.abs(weights) / 3.0, 0, 1), alpha=0.55, zorder=1)
        ax.add_collection(self.edge_lines)

        self.node_keys = list(positions)
        xy = np.array([positions[k] for k in self.node_keys])
        colours = [INPUT_COLOUR if k in inputs else OUTPUT_COLOUR if k in outputs
                   else UNUSED_COLOUR if k in unused else HIDDEN_COLOUR for k in self.node_keys]
        sizes = [90 if k in inputs else 260 if k in outputs else 150 for k in self.node_keys]
        self.nodes = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colours, edgecolors="#333333",
                                linewidths=0.8, zorder=3)

        for key, label in zip(inputs, INPUT_LABELS):
            x, y = positions[key]
            ax.text(x - 0.03, y, label, ha="right", va="center", fontsize=7.5)
        self.output_texts = []
        for key, label in zip(outputs, OUTPUT_LABELS):
            x, y = positions[key]
            self.output_texts.append(
                ax.text(x + 0.04, y, label, ha="left", va="center", fontsize=9, weight="bold"))

        hidden = len([k for k in genome.nodes if k not in outputs])
        disabled = sum(not c.enabled for c in genome.connections.values())
        ax.set_title(f"{info}\n{hidden} hidden nodes, {len(self.edges)} connections ({disabled} disabled)"
                     "   green = positive weight, red = negative", fontsize=9)
        if self.history_ax is not None and history is not None:
            self._draw_history(history)
        self.fig.tight_layout()
        self.refresh()

    def _draw_history(self, history):
        ax = self.history_ax
        ax.clear()
        if history:
            generations, best = zip(*history)
            ax.plot(generations, best, color="#4a6fa5", marker="o", markersize=3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("generation", fontsize=8)
        ax.set_ylabel("best fitness", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # ---- Live signals ---------------------------------------------------------
    def show_activations(self, values, config):
        """Colour nodes by their current value and connections by the signal they carry."""
        if not self.node_keys:
            return
        node_values = np.array([values.get(k, 0.0) for k in self.node_keys])
        self.nodes.set_facecolor(self.cmap((np.clip(node_values, -1, 1) + 1) / 2))
        if self.edges:
            signal = np.array([values.get(i, 0.0) * w for (i, _), w in self.edges])
            colours = self.cmap((np.clip(signal, -1.5, 1.5) / 1.5 + 1) / 2)
            colours[:, 3] = 0.15 + 0.85 * np.clip(np.abs(signal), 0, 1)
            self.edge_lines.set_color(colours)
        for text, key, label in zip(self.output_texts, config.genome_config.output_keys, OUTPUT_LABELS):
            text.set_text(f"{label}  {values.get(key, 0.0):+.2f}")
        self.refresh()


def _layout(genome, inputs, outputs):
    """Node positions in [0, 1]^2: inputs in column 0, hidden by depth, outputs last.

    Returns (positions, unused hidden nodes that don't affect any output).
    """
    enabled = [c.key for c in genome.connections.values() if c.enabled]
    layers, _ = feed_forward_layers(inputs, outputs, enabled)

    column = {k: 0 for k in inputs}
    for depth, layer in enumerate(layers, start=1):
        for node in layer:
            if node not in outputs:
                column[node] = depth
    hidden_columns = max([c for k, c in column.items() if k not in inputs] or [0])
    unused = {k for k in genome.nodes if k not in column and k not in outputs}
    if unused:
        hidden_columns = max(hidden_columns, 1)
        for k in unused:
            column[k] = max(1, (hidden_columns + 1) // 2)
    for k in outputs:
        column[k] = hidden_columns + 1

    by_column = {}
    for k, c in column.items():
        by_column.setdefault(c, []).append(k)
    positions = {}
    for c, nodes in by_column.items():
        nodes.sort(key=lambda k: inputs.index(k) if k in inputs else
                   outputs.index(k) if k in outputs else k)
        if c == 0:
            ys = np.linspace(1, 0, len(nodes))
        elif len(nodes) == 1:
            ys = [0.5]
        else:
            spread = min(1.0, 0.12 * len(nodes))
            ys = np.linspace(0.5 + spread / 2, 0.5 - spread / 2, len(nodes))
        for k, y in zip(nodes, ys):
            positions[k] = (c / (hidden_columns + 1), y)
    return positions, unused
