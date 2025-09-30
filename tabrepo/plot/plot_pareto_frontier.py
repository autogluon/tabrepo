from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_stats(df, on: str, groupby="method", method=["mean", "median", "std"]):
    return df[[groupby, on]].groupby(groupby).agg(method)[on]


def get_pareto_frontier(
    Xs,
    Ys,
    names=None,            # ← new
    *,
    max_X=True,
    max_Y=True,
    include_boundary_edges=True,
):
    """
    Compute the (piece‑wise constant) Pareto frontier and, in parallel,
    return the label associated with each frontier vertex.

    Parameters
    ----------
    Xs, Ys : Sequence[float]
        Coordinates of the points to consider.
    names : Sequence[str] or None, optional
        Label for each (X, Y) pair – e.g. `data["method"]`.
        If omitted, a list of ``None`` is used.
    max_X, max_Y : bool, default True
        If True the frontier favours larger values on that axis,
        otherwise it favours smaller values.
    include_boundary_edges : bool, default True
        If True, will include pareto front edges to the worst x and y values observed.

    Returns
    -------
    pareto_front : list[tuple[float, float]]
        The vertices that define the frontier, including the vertical
        “drop” segments needed for a step‑like plot.
    pareto_names : list[str | None]
        A label for each element in ``pareto_front``.
        Entries corresponding to the artificially inserted vertical
        drops are ``None``.
    """
    if names is None:
        names = [None] * len(Xs)
    if not (len(Xs) == len(Ys) == len(names)):
        raise ValueError("Xs, Ys and names must have the same length")

    # Sort primarily by X (descending if we maximise), secondarily by Y.
    pts = sorted(
        zip(Xs, Ys, names),
        key=lambda t: (t[0], t[1]),
        reverse=max_X,
    )

    pareto_front = [(pts[0][0], pts[0][1])]
    pareto_names = [pts[0][2]]
    best_y = pts[0][1]
    worst_y = pts[0][1]

    for x, y, label in pts[1:]:
        is_better = (y >= best_y) if max_Y else (y <= best_y)

        if is_better:
            # vertical segment to keep the frontier piece‑wise constant in X
            pareto_front.append((x, best_y))
            pareto_names.append(None)
            pareto_front.append((x, y))
            pareto_names.append(label)
            best_y = y

        is_worst = (y < worst_y) if max_Y else (y > worst_y)
        if is_worst:
            worst_y = y

    if include_boundary_edges:
        # add final horizontal segment to the worst point on X‑axis
        pareto_front.append((pts[-1][0], best_y))
        pareto_names.append(None)

        # add final vertical segment to the worst point on Y‑axis
        pareto_front.insert(0, (pts[0][0], worst_y))
        pareto_names.insert(0, None)

    return pareto_front, pareto_names


def plot_pareto(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    title: str,
    palette='Paired',
    hue: str = "Method",
    max_X: bool = False,
    max_Y: bool = True,
    sort_y: bool = False,
    ylim=None,
    save_path: str = None,
    add_optimal_arrow: bool = True,
    show: bool = True,
):
    if sort_y:
        data_order = data.sort_values(by=y_name, ascending=False)
        data = data.sort_values(by=y_name, ascending=max_Y)
        hue_order = list(data_order[hue])

        legend_order = list(data[hue])

        base = sns.color_palette(palette, n_colors=len(legend_order))
        if not max_Y:
            palette = base[::-1]
        else:
            palette = base
    else:
        hue_order = None

    g = sns.relplot(
        x=x_name,
        y=y_name,
        data=data,
        palette=palette,
        hue=hue,
        hue_order=hue_order,
        height=10,
        s=300,
    )

    Xs = list(data[x_name])
    Ys = list(data[y_name])
    labels = list(data[hue])
    pareto_front, pareto_names = get_pareto_frontier(Xs=Xs, Ys=Ys, names=labels, max_X=max_X, max_Y=max_Y)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)

    # ------------------------------------------------------------------
    # Label every real vertex on the Pareto frontier
    # ------------------------------------------------------------------
    import matplotlib.transforms as mtrans

    ax = plt.gca()                                  # current Axes
    offset_pts = 5                                  # ± points to nudge labels

    for (x, y), label in zip(pareto_front, pareto_names):
        if label is None:                           # skip dummy vertices
            continue

        # Decide which quadrant to nudge the text toward so labels sit
        # “outside” the frontier step.
        dx = offset_pts if max_X else -offset_pts
        dy = offset_pts if max_Y else -offset_pts
        ha = 'left' if max_X else 'right'
        va = 'bottom' if max_Y else 'top'

        # Use Axes.annotate with textcoords='offset points' so the offset
        # is interpreted in display points, independent of axis scaling.
        ax.annotate(
            label,
            xy=(x, y),                  # data point
            xytext=(dx, dy),            # offset in points
            textcoords='offset points',
            ha=ha,
            va=va,
            fontsize=9,
        )

    if ylim is not None:
        plt.ylim(ylim)
    g.set(xscale="log")
    fig = g.fig

    plt.grid()

    if add_optimal_arrow:
        ax = g.ax

        best_low_x = not max_X
        best_low_y = not max_Y

        corner_x = 0 if best_low_x else 1  # 0 = left, 1 = right  in axes fraction
        corner_y = 0 if best_low_y else 1  # 0 = bottom, 1 = top  in axes fraction

        # ------------------------------------------------------------
        # Arrow coordinates in **axes‑fraction space**
        offset = 0.10  # 10% in from the outer edge
        start = (
            corner_x + (+offset if corner_x == 0 else -offset),
            corner_y + (+offset if corner_y == 0 else -offset),
        )
        end = (corner_x, corner_y)  # the actual “best” corner

        # ------------------------------------------------------------
        # Draw a wide, filled green arrow (10% inset, axes‑fraction coords)
        arrow = ax.annotate(
            "", xy=end, xytext=start,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="Fancy,head_length=0.42,head_width=0.30,tail_width=0.30",
                facecolor="forestgreen",
                edgecolor="forestgreen",
                linewidth=0,  # no outline
                mutation_scale=100  # scales the sizes above
            ),
        )

        # ------------------------------------------------------------
        # Place readable text *inside* the arrow
        vec = np.array(end) - np.array(start)  # arrow vector (axes fraction)
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        if angle < -90 or angle > 90:  # keep text left→right
            angle += 180

        mid = (np.array(start) + np.array(end)) / 2  # midpoint of arrow

        ax.text(
            mid[0], mid[1], "Optimal",
            transform=ax.transAxes,
            rotation=angle, rotation_mode="anchor",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white",
        )

    # Add a title to the Figure
    fig.suptitle(title, fontsize=14)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_pareto_aggregated(
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    data_x: pd.DataFrame = None,
    x_method: str = "median",
    y_method: str = "mean",
    max_X: bool = False,
    max_Y: bool = True,
    ylim=(None, 1),
    hue: str = "Method",
    title: str = None,
    save_path: str = None,
    show: bool = True,
    include_method_in_axis_name: bool = True,
    sort_y: bool = False,
):
    if data_x is None:
        data_x = data
    if x_name not in data_x:
        raise AssertionError(f"Missing x_name='{x_name}' column in data_x")
    elif not is_numeric_dtype(data_x[x_name]):
        raise AssertionError(f"x_name='{x_name}' must be a numeric dtype")
    elif data_x[x_name].isnull().values.any():
        raise AssertionError(f"x_name='{x_name}' cannot contain NaN values")
    if y_name not in data:
        raise AssertionError(f"Missing y_name='{y_name}' column in data")
    elif not is_numeric_dtype(data[y_name]):
        raise AssertionError(f"y_name='{y_name}' must be a numeric dtype")
    elif data[y_name].isnull().values.any():
        raise AssertionError(f"y_name='{y_name}' cannot contain NaN values")
    y_vals = aggregate_stats(df=data, on=y_name, method=[y_method])[y_method]
    x_vals = aggregate_stats(df=data_x, on=x_name, method=[x_method])[x_method]
    if include_method_in_axis_name:
        x_name = f'{x_name} ({x_method})'
        y_name = f'{y_name} ({y_method})'
    df_aggregated = y_vals.to_frame(name=y_name)
    df_aggregated[x_name] = x_vals
    df_aggregated[hue] = df_aggregated.index

    plot_pareto(
        data=df_aggregated,
        x_name=x_name,
        y_name=y_name,
        title=title,
        save_path=save_path,
        max_X=max_X,
        max_Y=max_Y,
        ylim=ylim,
        hue=hue,
        sort_y=sort_y,
        show=show,
    )
