"""Plotly helpers. Every chart carries the 'Simulated data' label."""
from __future__ import annotations

import plotly.graph_objects as go

from .. import DATA_LABEL
from . import paths

TEMPLATE = "plotly_white"


def finalise(fig: go.Figure, title: str, height: int = 450) -> go.Figure:
    fig.update_layout(
        title=f"{title} | {DATA_LABEL}",
        template=TEMPLATE,
        height=height,
        margin=dict(l=60, r=30, t=70, b=60),
        annotations=list(fig.layout.annotations)
        + [dict(text=DATA_LABEL, xref="paper", yref="paper", x=1.0, y=1.08, showarrow=False,
                font=dict(size=11, color="#b00020"))],
    )
    return fig


def save_fig(fig: go.Figure, name: str, png: bool = True) -> None:
    out = paths.figures()
    fig.write_html(out / f"{name}__simulated_data.html", include_plotlyjs="cdn")
    if png:
        try:
            fig.write_image(out / f"{name}__simulated_data.png", scale=2)
        except Exception as exc:  # image export is optional
            (out / f"{name}__png_not_written.txt").write_text(str(exc)[:500])
