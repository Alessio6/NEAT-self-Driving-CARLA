"""Matplotlib window that draws a NEAT network and its live signals.

Used by train.py --viz (watch the structure evolve) and play.py --viz
(watch the trained network react to what the car sees).
"""

import matplotlib
import matplotlib.ticker
import numpy as np
from neat.graphs import feed_forward_layers

from carla_env import NUM_LIDAR_SECTORS

INPUT_LABELS = (
    [f"ray {round(i * 360 / NUM_LIDAR_SECTORS):>3}°" for i in range(NUM_LIDAR_SECTORS)]
    + ["speed", "heading err", "lane offset", "look-ahead"]
)
_DIRECTIONS = {0: "front", 90: "right", 180: "rear", 270: "left"}
INPUT_LABELS = [
    f"{label} {_DIRECTIONS[round(i * 360 / NUM_LIDAR_SECTORS)]}"
    if i < NUM_LIDAR_SECTORS and round(i * 360 / NUM_LIDAR_SECTORS) in _DIRECTIONS else label
    for i, label in enumerate(INPUT_LABELS)
]
OUTPUT_LABELS = ["steer", "throttle/brake"]

POSITIVE, NEGATIVE = "#2e9e5b", "#d6453d"


class NetworkView:
    def __init__(self, title="NEAT network", interactive=True, show_history=False, input_labels=None):
        self.input_labels = input_labels or INPUT_LABELS
        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        self.plt, self.LineCollection = plt, LineCollection
        self.interactive = interactive
        self.cmap = plt.get_cmap("coolwarm")

        if interactive:
            plt.ion()
        if show_history:
            self.fig, (self.ax, self.hist_ax) = plt.subplots(
                2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [4, 1]})
        else:
            self.fig, self.ax = plt.subplots(figsize=(9, 7))
            self.hist_ax = None
        self.fig.canvas.manager.set_window_title(title)
        self.genome_key = None
        if interactive:
            self.fig.show()

    # ---- Window state -----------------------------------------------------
    def is_open(self):
        return not self.interactive or self.plt.fignum_exists(self.fig.number)

    def refresh(self):
        if self.interactive:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path, dpi=110)

    def close(self):
        self.plt.close(self.fig)

    # ---- Structure --------------------------------------------------------
    def _layout(self, genome, config):
        gc = config.genome_config
        inputs, outputs = list(gc.input_keys), list(gc.output_keys)
        enabled = [cg.key for cg in genome.connections.values() if cg.enabled]
        layers, _ = feed_forward_layers(inputs, outputs, enabled)

        column = {k: 0 for k in inputs}
        for depth, layer in enumerate(layers, start=1):
            for node in layer:
                if node not in outputs:
                    column[node] = depth
        hidden_cols = max([c for n, c in column.items() if n not in inputs] or [0])
        unused = [n for n in genome.nodes if n not in column and n not in outputs]
        if unused:  # hidden nodes that don't affect the outputs
            hidden_cols = max(hidden_cols, 1)
            for n in unused:
                column[n] = max(1, (hidden_cols + 1) // 2)
        for n in outputs:
            column[n] = hidden_cols + 1

        by_col = {}
        for n, c in column.items():
            by_col.setdefault(c, []).append(n)
        pos = {}
        for c, nodes in by_col.items():
            nodes.sort(key=lambda k: (inputs.index(k) if k in inputs else
                                      outputs.index(k) if k in outputs else k))
            if c == 0:
                ys = np.linspace(1, 0, len(nodes))
            else:
                spread = min(1.0, 0.12 * len(nodes))
                ys = np.linspace(0.5 + spread / 2, 0.5 - spread / 2, len(nodes)) if len(nodes) > 1 else [0.5]
            for n, y in zip(nodes, ys):
                pos[n] = (c / (hidden_cols + 1), y)
        return pos, set(unused)

    def show_genome(self, genome, config, info="", history=None):
        """Redraw the network for `genome` (call once per new genome)."""
        gc = config.genome_config
        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(-0.32, 1.42)
        ax.set_ylim(-0.06, 1.1)

        pos, unused = self._layout(genome, config)
        self.pos = pos
        self.edges = [(cg.key, cg.weight) for cg in genome.connections.values()
                      if cg.enabled and cg.key[0] in pos and cg.key[1] in pos]
        segments = [(pos[i], pos[o]) for (i, o), _ in self.edges]
        weights = np.array([w for _, w in self.edges]) if self.edges else np.zeros(0)
        self.edge_widths = 0.4 + 2.6 * np.clip(np.abs(weights) / 3.0, 0, 1)
        colors = [(POSITIVE if w > 0 else NEGATIVE) for w in weights]
        self.edge_lc = self.LineCollection(segments, colors=colors, linewidths=self.edge_widths,
                                           alpha=0.55, zorder=1)
        ax.add_collection(self.edge_lc)

        self.node_keys = list(pos)
        xy = np.array([pos[k] for k in self.node_keys])
        face = []
        for k in self.node_keys:
            if k in gc.input_keys:
                face.append("#9db4d8")
            elif k in gc.output_keys:
                face.append("#f2b84b")
            elif k in unused:
                face.append("#dddddd")
            else:
                face.append("#b58ad6")
        sizes = [90 if k in gc.input_keys else 260 if k in gc.output_keys else 150 for k in self.node_keys]
        self.nodes_sc = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=face, edgecolors="#333333",
                                   linewidths=0.8, zorder=3)

        for i, k in enumerate(gc.input_keys):
            x, y = pos[k]
            label = self.input_labels[i] if i < len(self.input_labels) else f"input {i}"
            ax.text(x - 0.03, y, label, ha="right", va="center", fontsize=7.5)
        self.output_texts = []
        for i, k in enumerate(gc.output_keys):
            x, y = pos[k]
            self.output_texts.append(
                ax.text(x + 0.04, y, OUTPUT_LABELS[i], ha="left", va="center", fontsize=9, weight="bold"))

        n_hidden = len([k for k in genome.nodes if k not in gc.output_keys])
        n_disabled = sum(not cg.enabled for cg in genome.connections.values())
        ax.set_title(f"{info}\n{n_hidden} hidden nodes, {len(self.edges)} connections"
                     f" ({n_disabled} disabled)   green = positive weight, red = negative",
                     fontsize=9)

        if self.hist_ax is not None and history is not None:
            h = self.hist_ax
            h.clear()
            if history:
                gens, best = zip(*history)
                h.plot(gens, best, color="#4a6fa5", marker="o", markersize=3)
            h.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            h.set_xlabel("generation", fontsize=8)
            h.set_ylabel("best fitness", fontsize=8)
            h.tick_params(labelsize=7)
            h.grid(alpha=0.3)

        self.fig.tight_layout()
        self.refresh()

    # ---- Live signals -----------------------------------------------------
    def show_activations(self, values, config):
        """Colour nodes by their current value and connections by the signal they carry."""
        if not self.node_keys:
            return
        node_vals = np.array([values.get(k, 0.0) for k in self.node_keys])
        self.nodes_sc.set_facecolor(self.cmap((np.clip(node_vals, -1, 1) + 1) / 2))

        if self.edges:
            signal = np.array([values.get(i, 0.0) * w for (i, _), w in self.edges])
            colors = self.cmap((np.clip(signal, -1.5, 1.5) / 1.5 + 1) / 2)
            colors[:, 3] = 0.15 + 0.85 * np.clip(np.abs(signal), 0, 1)
            self.edge_lc.set_color(colors)

        for text, key, label in zip(self.output_texts, config.genome_config.output_keys, OUTPUT_LABELS):
            text.set_text(f"{label}  {values.get(key, 0.0):+.2f}")
        self.refresh()
