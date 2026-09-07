"""Standalone component and support summaries of recorded bootstrap draws."""

from pathlib import Path

import numpy as np


def plot_stability(arrays, metadata, directory):
    """Retain the requested replicate denominator when some refits fail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .configuration import file_identity
    from .production import write_json_exclusive
    from .statistics import summary

    rank, total = metadata["candidate"]["rank"], metadata["replicates"]
    if type(total) is not int or total < 1:
        raise ValueError("positive requested bootstrap count required")
    records = metadata["records"]
    if len(records) != total or [r["replicate"] for r in records] != list(range(total)):
        raise ValueError("bootstrap replicate records are incomplete or reordered")
    evaluated = np.array([r["status"] == "evaluated" for r in records])
    for key in ("signed_cosines", "principal_angles", "cluster_principal_angles"):
        if arrays[key].shape != (total, rank) or not np.isfinite(arrays[key][evaluated]).all():
            raise ValueError("bootstrap agreement arrays disagree with evaluated draws")
    reference = np.asarray(arrays["reference_feature_mask"])
    support = np.asarray(arrays["replicate_feature_masks"])
    if reference.ndim != 1 or reference.dtype.kind != "b" or support.shape != (total, len(reference)) or support.dtype.kind != "b":
        raise ValueError("bootstrap support dimensions disagree")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    identity = metadata["identity"]
    title = (f"SPW {identity['spw']} · block length {metadata['block_length']} · "
             f"{int(evaluated.sum())}/{total} evaluated draws")
    products = []
    numerical = {"requested_replicates": total, "evaluated_replicates": int(evaluated.sum()),
                 "comparison_space": metadata["comparison_space"], "components": []}

    def save(fig, name):
        path = directory / (name + ".png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        products.append(file_identity(path))

    def intervals(ax, values, ylabel):
        if not rank or not evaluated.any():
            ax.text(.5, .5, "No retained modes" if not rank else "No evaluated bootstrap draws",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            low, median, high = np.percentile(values[evaluated], [5, 50, 95], axis=0)
            ax.errorbar(np.arange(1, rank + 1), median, yerr=[median - low, high - median], fmt="o", capsize=3)
            ax.set_xticks(np.arange(1, rank + 1))
        ax.set(xlabel="Index", ylabel=ylabel)
        ax.grid(alpha=.2)

    aligned = np.abs(arrays["signed_cosines"])
    angles = np.rad2deg(arrays["principal_angles"])
    cluster_angles = np.rad2deg(arrays["cluster_principal_angles"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    intervals(axes[0], aligned, "Sign-aligned matched cosine")
    axes[0].set(xlabel="Matched component", ylim=(-.03, 1.03))
    intervals(axes[1], angles, "Principal angle (degrees)")
    axes[1].set(xlabel="Ordered subspace angle", ylim=(-2, 92))
    fig.suptitle(title + "\nMedian and 5th–95th percentiles of evaluated draws")
    save(fig, "bootstrap-agreement")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    intervals(ax, cluster_angles, "Within-cluster principal angle (degrees)")
    for cluster in metadata["spectral_clusters"]:
        ax.axvspan(min(cluster) + .65, max(cluster) + 1.35, alpha=.08, color="C0")
    ax.set(xlabel="Angle index within the declared spectral clusters", ylim=(-2, 92))
    boundary = metadata.get("cluster_crosses_rank_boundary")
    boundary_text = "unresolved cluster at retained-rank boundary" if boundary else "declared retained spectral clusters"
    if boundary is None:
        boundary_text = "discarded-direction spectrum unavailable"
    fig.suptitle(title + "\n" + boundary_text)
    save(fig, "bootstrap-clusters")

    counts = support[evaluated].sum(axis=0)
    fraction = counts / total
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    ax.plot(np.arange(len(reference)), fraction, ".", label="Evaluated draws with modeled support / requested draws")
    ax.plot(np.arange(len(reference)), reference.astype(float), "_", color="black", alpha=.4, label="Reference support")
    ax.set(xlabel="Feature index in the selected cylindrical view", ylabel="Fraction of requested draws",
           ylim=(-.03, 1.03))
    ax.legend(fontsize=8)
    fig.suptitle(title)
    save(fig, "bootstrap-support")
    for index in range(rank):
        numerical["components"].append({"index": index,
            "sign_aligned_cosine": summary(aligned[evaluated, index]),
            "ordered_subspace_angle_deg": summary(angles[evaluated, index]),
            "cluster_angle_deg": summary(cluster_angles[evaluated, index])})
    numerical["feature_support_counts"] = counts.tolist()
    numerical["failed_draws"] = [record for record in records if record["status"] != "evaluated"]
    write_json_exclusive(directory / "figures.json", {"identity": identity, "figures": products,
                         "bootstrap_measurements": numerical})
    return products
