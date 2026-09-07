"""Standalone coordinate and mode-energy comparisons between spectral windows."""

from pathlib import Path

import numpy as np


def plot_cross_spw(arrays, metadata, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .configuration import file_identity
    from .production import write_json_exclusive
    from .splits import continuous_segments

    if metadata.get("purpose") != "cross_spw_diagnostics":
        raise ValueError("cross-spectrum diagnostic product required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    written = []
    left, right = (metadata[label]["identity"]["spw"] for label in ("left", "right"))
    title = f"SPW {left} versus SPW {right}"

    def save(fig, name):
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(file_identity(path))

    comparison = metadata["mode_similarity"]
    for method, label in (("exact", "Exact native centers"), ("rebinned", "Common delay bins")):
        report = comparison.get(method, {})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        for ax, quantity, description in zip(axes, ("signed_similarity", "squared_loading_similarity"),
                                             ("Signed component cosine", "Squared-loading cosine")):
            if report.get("available"):
                values = arrays[method + "_" + quantity]
                image = ax.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r" if quantity.startswith("signed") else "viridis",
                                  vmin=-1 if quantity.startswith("signed") else 0, vmax=1)
                ax.set(xticks=np.arange(values.shape[1]), xticklabels=np.arange(1, values.shape[1] + 1),
                       yticks=np.arange(values.shape[0]), yticklabels=np.arange(1, values.shape[0] + 1),
                       xlabel=f"SPW {right} component", ylabel=f"SPW {left} component", title=description)
                fig.colorbar(image, ax=ax, label="Cosine similarity")
            else:
                ax.text(.5, .5, report.get("reason", comparison.get("reason", "Comparison unavailable")),
                        ha="center", va="center", wrap=True, transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
        support = f" · {report['matched_cells']} shared cells" if report.get("available") else ""
        fig.suptitle(title + " · " + label + support + "\nDescriptive representation loadings; arbitrary mode signs")
        save(fig, method + "-similarity")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, side in zip(axes, ("left", "right")):
        values = arrays.get(side + "_mode_high_k_energy_fraction")
        description = metadata[side]
        if values is None or not len(values):
            reason = description.get("physical_mode_energy", {}).get("reason", "Rank-zero model: no mode-energy fractions")
            ax.text(.5, .5, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
        else:
            times = np.unwrap(arrays[side + "_lst_rad"]) * 12 / np.pi
            segments = continuous_segments(arrays[side + "_window_ids"], np.arange(len(times)))
            for mode, fractions in enumerate(values):
                for index, segment in enumerate(segments):
                    ax.plot(times[segment], fractions[segment], ".-", color=f"C{mode % 10}",
                            markersize=2, linewidth=.7, label=f"Mode {mode + 1}" if index == 0 else None)
            ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value % 24:g}"))
        ax.set(xlabel="Physical LST (hours, continuous ordering)", ylabel="Fraction of conditional mode energy",
               ylim=(-.03, 1.03), title=f"SPW {description['identity']['spw']} · rank {description['rank']}")
    fig.suptitle(title + " · mode power above 0.3 h Mpc⁻¹\nDescriptive physical-power contrasts; no signal-preservation test")
    save(fig, "high-k-mode-energy")
    write_json_exclusive(directory / "figures.json", {"purpose": metadata["purpose"], "figures": written})
    return written
