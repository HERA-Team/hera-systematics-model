"""Standalone scientific figures from verified numerical diagnostic products."""

from pathlib import Path

import numpy as np


def plot_diagnostics(arrays, metadata, directory):
    """Render signed maps, score distributions and physical-time validation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import FuncFormatter

    from .configuration import file_identity
    from .production import write_json_exclusive
    from .splits import continuous_segments

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    written = []
    identity = metadata["identity"]
    label = f"SPW {identity['spw']} · {identity['polarization']}"
    units = identity["power_units"]
    shape = arrays["mean_residual"].shape

    def save(fig, name):
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(file_identity(path))

    def plane(ax, values, title, norm=None, cmap="viridis"):
        artist = ax.pcolormesh(arrays["kperp"], arrays["kparallel"], np.asarray(values).T,
                              shading="nearest", cmap=cmap, norm=norm, rasterized=True)
        ax.set(xlabel=r"$k_\perp$ ($h$ Mpc$^{-1}$)", ylabel=r"$k_\parallel$ ($h$ Mpc$^{-1}$)", title=title)
        return artist

    def signed_norm(values):
        finite = np.asarray(values)[np.isfinite(values)]
        maximum = float(np.max(np.abs(finite))) if len(finite) else 0.
        maximum = maximum if maximum > 0 else 1.
        return SymLogNorm(linthresh=maximum / 1000., vmin=-maximum, vmax=maximum)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    means = [arrays[f"mean_{name}"] for name in ("corrupted", "ideal", "residual")]
    norm = signed_norm(np.stack(means))
    for ax, values, title in zip(axes, means, ("Mean corrupted", "Mean ideal", "Mean residual")):
        image = plane(ax, values, title, norm, "RdBu_r")
    fig.colorbar(image, ax=axes, label=units)
    fig.suptitle(label + " · measured means on common valid support")
    save(fig, "mean-planes")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    maps = [arrays["valid_fraction"], arrays["contributor_counts"].mean(axis=(0, 3)), arrays["median_noise"]]
    for ax, values, title, unit in zip(axes, maps,
            ("Valid window fraction", "Mean contributors per delay side", "Median corrupted noise"),
            ("Fraction", "Baseline count", units)):
        image = plane(ax, values, title)
        fig.colorbar(image, ax=ax, label=unit)
    fig.suptitle(label)
    save(fig, "support-planes")

    if "components" in arrays:
        for index, component in enumerate(arrays["components"]):
            fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
            image = plane(ax, component.reshape(shape), f"{label} · signed component {index + 1}",
                          signed_norm(component), "RdBu_r")
            fig.colorbar(image, ax=ax, label=f"Loading in {metadata['selected']['representation']} representation")
            save(fig, f"component-{index + 1:02d}")
    if "singular_values" in arrays:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        values = arrays["singular_values"]
        axes[0].plot(np.arange(1, len(values) + 1), values, ".-")
        axes[0].set(xlabel="Component count", ylabel="Singular value", title="Scree")
        if len(arrays["explained_variance_ratio"]):
            axes[1].plot(np.arange(1, len(values) + 1), arrays["cumulative_variance_ratio"], ".-")
            axes[1].set(xlabel="Component count", ylabel="Cumulative variance fraction", ylim=(0, 1.02))
        else:
            axes[1].text(.5, .5, "Ordinary PCA variance fractions\nare unavailable for this method",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_axis_off()
        fig.suptitle(label + " · descriptive fit")
        save(fig, "scree")

    lst = arrays["lst_unwrapped_hours"]
    segments = continuous_segments(arrays["window_ids"], np.arange(len(lst)))
    formatter = FuncFormatter(lambda value, _: f"{value % 24:g}")
    if metadata.get("physical_mode_energy", {}).get("available"):
        for mode, mean_squared in enumerate(arrays["mode_mean_squared_contrast"]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
            image = plane(axes[0], np.sqrt(mean_squared).reshape(shape), "RMS one-mode residual contrast")
            fig.colorbar(image, ax=axes[0], label=units)
            for region, label_region, color in (("full", "Full modeled plane", "C0"),
                                                 ("high_k", r"$k_\parallel > 0.3\,h$ Mpc$^{-1}$", "C1")):
                rms = np.sqrt(arrays[f"mode_{region}_window_energy"][mode])
                for index, segment in enumerate(segments):
                    axes[1].plot(lst[segment], rms[segment], ".-", color=color,
                                 label=label_region if index == 0 else None)
            axes[1].xaxis.set_major_formatter(formatter)
            axes[1].set(xlabel="Physical LST (hours)", ylabel=f"RMS residual contrast ({units})")
            axes[1].legend(fontsize=8)
            fig.suptitle(f"{label} · mode {mode + 1} · conditional decoder response")
            save(fig, f"physical-mode-{mode + 1:02d}")
    for mode in range(arrays["scores"].shape[1]):
        values = arrays["scores"][:, mode]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for segment in segments:
            axes[0, 0].plot(lst[segment], values[segment], ".-", color="C0")
        axes[0, 0].xaxis.set_major_formatter(formatter)
        axes[0, 0].set(xlabel="Physical LST (hours, continuous ordering)", ylabel="Descriptive score")
        axes[0, 1].hist(values, bins="auto", density=True)
        axes[0, 1].set(xlabel="Descriptive score", ylabel="Density")
        if f"qq_theoretical_{mode}" in arrays:
            x, y = arrays[f"qq_theoretical_{mode}"], arrays[f"qq_observed_{mode}"]
            axes[1, 0].plot(x, y, ".")
            axes[1, 0].plot(x, x, "k--", linewidth=.8)
        axes[1, 0].set(xlabel="Normal quantile", ylabel="Standardized score quantile")
        acf = arrays[f"acf_{mode}"]
        axes[1, 1].plot(np.arange(len(acf)), acf, ".-")
        axes[1, 1].set(xlabel="Exact native-window lag", ylabel="Autocorrelation")
        fig.suptitle(f"{label} · mode {mode + 1} · one correlated LST arc")
        save(fig, f"scores-{mode + 1:02d}")

    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for name, legend, color in (("window_loss", "Selected model", "C0"), ("zero_baseline_loss", "Zero residual", "C1"),
                                ("mean_baseline_loss", "Training mean", "C2")):
        for index, segment in enumerate(segments):
            ax.plot(lst[segment], arrays[name][segment], ".-", color=color, label=legend if index == 0 else None)
    finite_loss = arrays["mean_baseline_loss"][np.isfinite(arrays["mean_baseline_loss"])]
    threshold = max(float(np.median(finite_loss)) * 1e-4, 1e-20) if len(finite_loss) else 1e-20
    ax.set_yscale("symlog", linthresh=threshold)
    ax.xaxis.set_major_formatter(formatter)
    ax.set(xlabel="Physical LST (hours, continuous ordering)", ylabel=f"Weighted MSE ({units})²",
           title=label + " · withheld-cell prediction on outer time folds")
    ax.legend()
    save(fig, "predictive-loss")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    candidates = metadata.get("candidates", [])
    colors = {"zero": "black", "mean": "gray", "complete": "C0", "masked": "C1", "kernel": "C2"}
    markers = {"linear": "o", "noise_weighted": "s", "signed_asinh": "^", "log_ratio": "x"}
    for index, ax in enumerate(axes.ravel()):
        losses = arrays.get(f"inner_losses_{index}")
        if losses is None:
            report = metadata.get("folds", [])[index]
            frozen = report.get("inner", {}).get("selection_repeated") is False
            ax.text(.5, .5, "Primary configuration held fixed" if frozen else "Inner selection unavailable",
                    ha="center", transform=ax.transAxes)
            continue
        finite = np.isfinite(losses).all(axis=1)
        for method, color in colors.items():
            for representation, marker in markers.items():
                use = [i for i, c in enumerate(candidates) if c["method"] == method and c["representation"] == representation and finite[i]]
                if use:
                    ax.scatter([candidates[i]["rank"] for i in use], losses[use].mean(axis=1),
                               c=color, marker=marker, alpha=.45, s=16, label=f"{method} / {representation}")
        positive = losses[np.isfinite(losses) & (losses > 0)]
        ax.set_yscale("symlog", linthresh=float(np.median(positive)) * 1e-6 if len(positive) else 1e-20)
        ax.set(xlabel="Rank", ylabel="Mean inner time-fold predictive loss", title=f"Outer fold {index + 1}")
    handles = {}
    for ax in axes.ravel():
        hs, ls = ax.get_legend_handles_labels()
        handles.update(zip(ls, hs))
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(label + " · candidate losses; each point is one configuration")
    save(fig, "rank-selection")

    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    eligible = arrays["eligible"].sum(axis=1)
    for name in ("modeled", "mean_only", "zero_only", "unavailable", "excluded"):
        fraction = np.divide(arrays[name].sum(axis=1), eligible, out=np.full(len(lst), np.nan), where=eligible > 0)
        ax.plot(lst, fraction, ".", label=name.replace("_", " "))
    ax.xaxis.set_major_formatter(formatter)
    ax.set(xlabel="Physical LST (hours, continuous ordering)", ylabel="Fraction of eligible plane", ylim=(-.03, 1.03), title=label + " · prediction coverage")
    ax.legend(ncol=3)
    save(fig, "coverage")
    write_json_exclusive(directory / "figures.json", {"identity": identity, "figures": written})
    return written
