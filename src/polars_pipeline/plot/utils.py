from pathlib import Path

from matplotlib.figure import Figure


def log_figure(fig: Figure, caption: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_path = log_dir / f"{caption.replace(" ", "_")}.png"
    fig.savefig(fig_path)
