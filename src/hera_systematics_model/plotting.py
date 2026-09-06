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
    for name, legend in (("window_loss", "Selected model"), ("zero_baseline_loss", "Zero residual"),
                         ("mean_baseline_loss", "Training mean")):
        for index, segment in enumerate(segments):
            ax.plot(lst[segment], arrays[name][segment], ".-", label=legend if index == 0 else None)
    ax.set_yscale("symlog", linthresh=max(float(np.nanmedian(arrays["mean_baseline_loss"])) * 1e-4, 1e-20))
    ax.xaxis.set_major_formatter(formatter)
    ax.set(xlabel="Physical LST (hours, continuous ordering)", ylabel=f"Weighted MSE ({units})²",
           title=label + " · withheld-cell prediction on outer time folds")
    ax.legend()
    save(fig, "predictive-loss")

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
