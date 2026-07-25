#!/usr/bin/env python3
"""Assemble the H6C-IDR2 inventory manifest (JSON + CSV).

Inputs: the JSONL metadata dump (06_metadata.jsonl) plus counts/sizes recorded
during the read-only reconnaissance (see inventory_commands.sh).
Outputs: h6c_idr2_inventory.json and h6c_idr2_inventory.csv.
"""
import csv
import json
import sys
from pathlib import Path

SCRATCH = Path(__file__).parent
OUT_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRATCH / "h6c_idr2_inventory.json"
OUT_CSV = OUT_JSON.with_suffix(".csv")

BASE = "/lustre/aoc/projects/hera/Validation/DataReleases/H6C-IDR2"
STORE = "/home/herastore02-2/Validation/H6C-IDR2"  # symlink target of BASE entries
WORK = "/lustre/aoc/projects/hera/Validation/H6C_IDR2"

meta = {}
jsonl = SCRATCH / "06_metadata.jsonl"
if jsonl.exists():
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rec = json.loads(line)
            meta[rec["label"]] = rec


def m(label, *keys):
    rec = meta.get(label, {})
    out = {}
    for k in keys:
        if k in rec:
            out[k] = rec[k]
    return out

UVH5_KEYS = ("Ntimes", "Nfreqs", "Nbls", "Npols", "Nants_data", "vis_units",
             "freq_range_hz", "time_range_jd", "lst_range_rad",
             "integration_time_s", "channel_width_hz")

products = [
    # ---------------- release snapshot: ideal sims ----------------
    dict(product_id="chunked_ideal_eor", role="ideal_sim",
         path=f"{BASE}/chunked-ideal-sims/eor-grf-1024",
         file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.eor-grf-1024.uvh5",
         sky_components="eor_21cm_grf_nside1024", corruptions="none (beam only)",
         generator="hera_sim FFTVis (pyuvdata 3.1.1.dev7)",
         sample_metadata=m("chunked_ideal_eor", *UVH5_KEYS)),
    dict(product_id="chunked_ideal_gsm", role="ideal_sim",
         path=f"{BASE}/chunked-ideal-sims/gsm_nside1024",
         file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.gsm_nside1024.uvh5",
         sky_components="diffuse_gsm08_nside1024", corruptions="none (beam only)",
         generator="hera_sim FFTVis (pyuvdata 3.0.1.dev98)",
         sample_metadata=m("chunked_ideal_gsm", *UVH5_KEYS)),
    dict(product_id="chunked_ideal_ptsrc", role="ideal_sim",
         path=f"{BASE}/chunked-ideal-sims/ptsrc1024",
         file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.ptsrc1024.uvh5",
         sky_components="ptsrc_ateam_plus_uniform", corruptions="none (beam only)",
         generator="hera_sim FFTVis (pyuvdata 3.2.0, regenerated 2026-03)",
         sample_metadata=m("chunked_ideal_ptsrc", *UVH5_KEYS)),
    dict(product_id="chunked_ideal_ptsrc_deprecated", role="deprecated",
         path=f"{BASE}/chunked-ideal-sims/ptsrc1024-original-wrong-specindex",
         file_format="uvh5", n_files=8641,
         filename_convention="zen.LST.{lst:9.7f}.ptsrc1024.uvh5",
         sky_components="ptsrc_ateam_plus_uniform", corruptions="none (beam only)",
         generator="hera_sim FFTVis",
         sample_metadata={}),
    # ---------------- release snapshot: mock input components ----------------
    dict(product_id="sim_data_eor", role="mock_component",
         path=f"{BASE}/sim_data/eor", file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.eor.uvh5",
         sky_components="eor_21cm",
         corruptions="none in file; systematics added downstream per-night",
         generator="hera_sim FFTVis, downselected to core antennas",
         sample_metadata=m("sim_data_eor", *UVH5_KEYS)),
    dict(product_id="sim_data_diffuse", role="mock_component",
         path=f"{BASE}/sim_data/diffuse", file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.diffuse.uvh5",
         sky_components="diffuse_gsm08",
         corruptions="none in file; systematics added downstream per-night",
         generator="hera_sim FFTVis, downselected to core antennas",
         sample_metadata=m("sim_data_diffuse", *UVH5_KEYS)),
    dict(product_id="sim_data_ptsrc", role="mock_component",
         path=f"{BASE}/sim_data/ptsrc", file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.ptsrc.uvh5",
         sky_components="ptsrc_ateam_plus_uniform",
         corruptions="none in file; systematics added downstream per-night",
         generator="hera_sim FFTVis, downselected to core antennas",
         sample_metadata={}),
    dict(product_id="sim_data_foregrounds", role="mock_component",
         path=f"{BASE}/sim_data/foregrounds", file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.foregrounds.uvh5",
         sky_components="diffuse + ptsrc",
         corruptions="none in file; systematics added downstream per-night",
         generator="hera_sim FFTVis, downselected to core antennas",
         sample_metadata=m("sim_data_foregrounds", *UVH5_KEYS)),
    dict(product_id="sim_data_sum", role="mock_component",
         path=f"{BASE}/sim_data/sum", file_format="uvh5", n_files=8640,
         filename_convention="zen.LST.{lst:9.7f}.sum.uvh5",
         sky_components="eor + diffuse + ptsrc",
         corruptions="none in file; systematics added downstream per-night",
         generator="hera_sim FFTVis, downselected to core antennas",
         sample_metadata=m("sim_data_sum", *UVH5_KEYS)),
    # ---------------- sky models ----------------
    dict(product_id="sky_canonical_eor", role="sky_model",
         path=f"{BASE}/sky_models/Canonical/eor-grf-1024", file_format="skyh5",
         n_files=1536, filename_convention="fch{ch:04d}.skyh5",
         sky_components="eor_21cm_grf healpix nside=1024 (1 freq/file)",
         corruptions="n/a", generator="pyradiosky",
         sample_metadata=m("sky_model_eor_chunk", "Ncomponents", "nside",
                           "component_type", "spectral_type")),
    dict(product_id="sky_canonical_gsm", role="sky_model",
         path=f"{BASE}/sky_models/Canonical/gsm_nside1024", file_format="skyh5",
         n_files=1536, filename_convention="fch{ch:04d}.skyh5",
         sky_components="gsm08 diffuse healpix nside=1024", corruptions="n/a",
         generator="pyradiosky", sample_metadata={}),
    dict(product_id="sky_canonical_ptsrc", role="sky_model",
         path=f"{BASE}/sky_models/Canonical/ptsrc1024", file_format="skyh5",
         n_files=1, filename_convention="full.skyh5",
         sky_components="12,543,077 point sources, spectral_index type",
         corruptions="n/a", generator="pyradiosky",
         sample_metadata=m("sky_model_ptsrc_full", "Ncomponents",
                           "component_type", "spectral_type")),
    dict(product_id="sky_canonical_ptsrc_deprecated", role="deprecated",
         path=f"{BASE}/sky_models/Canonical/ptsrc1024-original-wrong-specidx",
         file_format="skyh5", n_files=1536, filename_convention="fch{ch:04d}.skyh5",
         sky_components="point sources (superseded spectral model)",
         corruptions="n/a", generator="pyradiosky", sample_metadata={}),
    dict(product_id="sky_raw", role="sky_model",
         path=f"{BASE}/sky_models/raw", file_format="skyh5/h5", n_files=2,
         filename_convention="ateam.skyh5, covariance.h5",
         sky_components="A-Team catalog; 56 GB GRF covariance",
         corruptions="n/a", generator="", sample_metadata={}),
    dict(product_id="sky_extra", role="sky_model_extra",
         path=f"{BASE}/sky_models/Extra", file_format="skyh5", n_files=None,
         filename_convention="(8 subdirs: eor, eor-grf-1024-monopole, "
         "eor-grf-1024-only-monopole, eor-grf-512, gsm_nside128/256/512, ptsrc512)",
         sky_components="alternative resolutions/variants", corruptions="n/a",
         generator="pyradiosky", sample_metadata={}),
    # ---------------- lstbin outputs ----------------
    dict(product_id="lstbin_LST", role="lstbinned_mock",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint",
         file_format="uvh5", n_files=4088,
         filename_convention="zen.LST.{lst:7.5f}.000.sum.uvh5 (2044) + "
         "zen.LST.{lst:7.5f}.autos.sum.uvh5 (2044)",
         sky_components="eor + foregrounds",
         corruptions="mutual coupling + thermal noise (Trx=100K) + bandpass gains "
         "(8% spread), per-night seeds; then smooth_calibrated, red_avg, LST-binned "
         "with simultaneous inpainting (500 ns min delay) and lstcal",
         generator="hera_cal lst_stack (h6c_idr2_validation env); "
         "14 nights JD 2459861-2459876",
         sample_metadata=m("lstbin_LST", *UVH5_KEYS)),
    dict(product_id="lstbin_STD", role="lstbinned_mock_aux",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint",
         file_format="uvh5", n_files=4088,
         filename_convention="zen.STD.{lst:7.5f}.[000|autos].sum.uvh5",
         sky_components="(std over nights)", corruptions="as lstbin_LST",
         generator="hera_cal lst_stack", sample_metadata={}),
    dict(product_id="lstbin_HIGHZ", role="lstbinned_mock_aux",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint",
         file_format="h5 (indices,zsq)", n_files=2044,
         filename_convention="zen.HIGHZ.{lst:7.5f}.000.sum.uvh5",
         sky_components="n/a", corruptions="n/a",
         generator="lststack QA (save_metric_data): catalogs of frequency "
         "regions with Z^2 above the predicted 99.85th percentile or inpainted",
         sample_metadata={}),
    dict(product_id="lstbin_METRICS", role="lstbinned_mock_aux",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint",
         file_format="h5 (meta,metrics)", n_files=2044,
         filename_convention="zen.LSTBIN-METRICS.{lst:7.5f}.000.sum.uvh5",
         sky_components="n/a", corruptions="n/a",
         generator="lststack QA: cross-night Z^2 consistency reductions",
         sample_metadata={}),
    dict(product_id="lstbin_qa_notebooks", role="qa",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint",
         file_format="ipynb+html", n_files=2045,
         filename_convention="lststack.{chunk:04d}.000.ipynb/.html",
         sky_components="n/a", corruptions="n/a", generator="hera_opm makeflow",
         sample_metadata={}),
    dict(product_id="lstbin_configs", role="provenance",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint",
         file_format="toml/yaml/h5", n_files=4,
         filename_convention="lstbin-config.toml, lstavg-config.toml, "
         "environment.yaml, file-config.h5",
         sky_components="n/a", corruptions="n/a", generator="",
         sample_metadata={}),
    # ---------------- single-baseline products ----------------
    dict(product_id="single_baseline_uvh5", role="upstream_pspec_input",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/single_baseline_files",
         file_format="uvh5", n_files=884,
         filename_convention="zen.LST.baseline.{a}_{b}.sum.uvh5",
         sky_components="eor + foregrounds", corruptions="as lstbin_LST",
         generator="hera_cal lst_stack, rechunked per baseline; "
         "1 auto + 883 cross baselines, 4088 LSTs x 1536 freqs each",
         sample_metadata=m("single_baseline_uvh5", *UVH5_KEYS)),
    dict(product_id="single_baseline_pspec", role="upstream_pspec",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/single_baseline_files",
         file_format="pspec.h5 (PSpecContainer/UVPSpec)", n_files=789,
         filename_convention="zen.LST.baseline.{a}_{b}.sum.pspec.h5",
         sky_components="eor + foregrounds", corruptions="as lstbin_LST",
         generator="hera_pspec 0.4.3.dev65 via hnote "
         "(single_baseline_postprocessing_and_pspec, h6c_pspec_14band.toml); "
         "789 completed + 95 skipped (autos / fully flagged polarization); "
         "14 spws, 4 pol pairs, taper bh, norm I, units (mK)^2 h^-3 Mpc^3",
         sample_metadata=m("single_baseline_pspec", "pspec")),
    dict(product_id="mini_dataset", role="test_subset",
         path=f"{BASE}/lstbin-outputs/v3.1-simultaneous-inpaint/inpaint/mini_dataset",
         file_format="uvh5", n_files=5,
         filename_convention="zen.LST.{lst_hours}_hours.mini_dataset.sum.uvh5",
         sky_components="eor + foregrounds", corruptions="as lstbin_LST",
         generator="hera_cal lst_stack (downsampled); 36 times x 384 freqs x 884 bls",
         sample_metadata=m("mini_dataset", *UVH5_KEYS)),
    # ---------------- merged pspec products (working area) ----------------
    dict(product_id="fg_eor_sum_merged", role="upstream_pspec_merged",
         path=f"{WORK}/lstbin-outputs/redavg-smoothcal-inpaint-500ns-lstcal/"
              "inpaint/single_baseline_files/baselines_merged.pspec.h5",
         file_format="pspec.h5 (PSpecContainer/UVPSpec)", n_files=1,
         filename_convention="baselines_merged.pspec.h5",
         sky_components="eor + foregrounds", corruptions="as lstbin_LST",
         generator="MERGE_SINGLE_BASELINE_FILES makeflow action (2025-04); "
         "789 blpairs, 106549 bltpairs, 6177 times, 14 spws, 4 pol pairs, 31 GB",
         sample_metadata={}),
    dict(product_id="fg_eor_sum_merged_tavg", role="fim_direction_pspec",
         path=f"{WORK}/lstbin-outputs/redavg-smoothcal-inpaint-500ns-lstcal/"
              "inpaint/single_baseline_files/baselines_merged.tavg.pspec.h5",
         file_format="pspec.h5", n_files=1,
         filename_convention="baselines_merged.tavg.pspec.h5",
         sky_components="eor + foregrounds", corruptions="as lstbin_LST",
         generator="MERGE_SINGLE_BASELINE_FILES makeflow action (2025-04); "
         "time_and_interleave_averaged: 789 blpairs, 42 times, pI, 79 MB",
         sample_metadata={}),
    dict(product_id="eor_only_lstbin", role="lstbinned_mock_branch",
         path=f"{WORK}/lstbin-outputs/eor-only",
         file_format="uvh5 + pspec.h5", n_files=4045,
         filename_convention="zen.LST.*.sum.uvh5 + single_baseline_files/ + "
         "mini_dataset/",
         sky_components="eor only",
         corruptions="not determined from file metadata",
         generator="same lst_stack + single-baseline + pspec chain "
         "(789 completed + 95 skipped)",
         sample_metadata={}),
    dict(product_id="eor_only_merged", role="upstream_pspec_merged",
         path=f"{WORK}/lstbin-outputs/eor-only/single_baseline_files/"
              "baselines_merged.pspec.h5",
         file_format="pspec.h5", n_files=1,
         filename_convention="baselines_merged.pspec.h5",
         sky_components="eor only", corruptions="see eor_only_lstbin",
         generator="MERGE_SINGLE_BASELINE_FILES makeflow action (2025-04); "
         "789 blpairs, 106485 bltpairs, 5321 times, 14 spws, 4 pol pairs, 47 GB",
         sample_metadata={}),
    dict(product_id="eor_only_merged_tavg", role="fim_direction_pspec",
         path=f"{WORK}/lstbin-outputs/eor-only/single_baseline_files/"
              "baselines_merged.tavg.pspec.h5",
         file_format="pspec.h5", n_files=1,
         filename_convention="baselines_merged.tavg.pspec.h5",
         sky_components="eor only", corruptions="see eor_only_lstbin",
         generator="MERGE_SINGLE_BASELINE_FILES makeflow action (2025-04); "
         "time_and_interleave_averaged: 789 blpairs, 38 times, pI, 128 MB",
         sample_metadata={}),
    dict(product_id="makeflow_pspec", role="provenance",
         path=f"{WORK}/makeflow-pspec",
         file_format="toml + makeflow logs/wrappers", n_files=None,
         filename_convention="h6c_pspec_14band.toml + fg-eor-sum/ + eor-only/",
         sky_components="n/a", corruptions="n/a",
         generator="hera_opm makeflow, slurm, conda env h6c_idr2_validation; "
         "per-baseline wrappers + logs for both branches incl. MERGE logs",
         sample_metadata={}),
    # ---------------- working area (not in DataReleases) ----------------
    dict(product_id="work_nightly", role="pipeline_working_area",
         path=f"{WORK}/{{2459861..2459876}} (14 nights)",
         file_format="uvh5/calfits/h5/csv/html", n_files=27932,
         filename_convention="zen.{jd}.{sd}.sum.<stage>; 1862 files/stage/night "
         "(n_files counted for night 2459861)",
         sky_components="eor + foregrounds",
         corruptions="mutual coupling + noise + bandpass gains injected per night "
         "(configs/{jd}.yaml, hera_sim); raw zen.*.sum.uvh5 are 0-byte placeholders",
         generator="MOCK_DATA makeflow action (h6c_analysis_validation.mf -> "
         "do_MOCK_DATA.sh -> mock_data.py): reads ideal sim_data/sum, uses real "
         "H6C files as time/metadata reference, applies coupling -> noise -> "
         "bandpass via hera_sim.Simulator.add(); then full H6C calibration "
         "pipeline (omnical, smoothcal, red_avg)",
         sample_metadata={}),
    dict(product_id="work_configs", role="provenance",
         path=f"{WORK}/configs", file_format="yaml/npz", n_files=None,
         filename_convention="{jd}.yaml + hera_350_coupling_matrix.npz, omegas.npz, "
         "bandpass.npz",
         sky_components="n/a",
         corruptions="defines them: MutualCoupling (coupling matrix), thermal_noise "
         "(Trx=100K, seeded), bandpass_gain (8% spread, tanh band edges, per-night seed)",
         generator="hera_sim config YAMLs",
         sample_metadata={}),
]

manifest = {
    "_schema": {
        "product_id": "stable identifier for this inventory",
        "role": "ideal_sim | mock_component | sky_model | lstbinned_mock | "
                "upstream_pspec | provenance | deprecated | ...",
        "path": "absolute directory or file on NRAO storage",
        "file_format": "container format",
        "n_files": "file count at -maxdepth 1 (null = not counted)",
        "filename_convention": "naming pattern",
        "sky_components": "sky content",
        "corruptions": "systematics applied (and where)",
        "generator": "producing software/workflow, from file history and logs",
        "sample_metadata": "metadata extracted read-only from representative files",
    },
    "inventory_date": "2026-07-26",
    "method": "read-only inspection from herapost-master; see "
              "scripts/inventory/ for exact commands",
    "base_path_release": BASE,
    "base_path_release_storage_target": STORE,
    "base_path_working": WORK,
    "total_sizes": {
        "chunked-ideal-sims": "41T",
        "sim_data": "12T",
        "sky_models": "3.2T",
        "lstbin-outputs": "1.4T",
    },
    "environments": {
        "pipeline": "h6c_idr2_validation (environment.yaml in "
                    "lstbin-outputs/v3.1-simultaneous-inpaint)",
        "analysis": "validation_env @ /lustre/aoc/projects/hera/kmandar/"
                    "miniconda3 (python 3.10.14, h5py 3.11.0, pyuvdata 3.2.0, "
                    "hera_pspec 0.4.3.dev89, hera_cal 3.7.4)",
    },
    "data_facts": [
        "884 single-baseline uvh5 exist (1 auto + 883 cross); the pspec stage "
        "is complete per the makeflow logs: 789 completed + 95 deliberately "
        "skipped (autos / fully flagged polarization).",
        "The official merged products are baselines_merged.pspec.h5 and "
        "baselines_merged.tavg.pspec.h5 (both branches, 2025-04); an older "
        "custom merge (baselines_merged_131times.pspec.h5, 776 blpairs x 131 "
        "ordinal times) is superseded.",
        "The DataReleases snapshot lacks the merged pspec products and the "
        "eor-only branch; those exist only in the Lustre working area.",
        "sim_data mock components contain no injected systematics per their "
        "file histories; corruption (mutual coupling, thermal noise, bandpass "
        "gains) is applied downstream per night via configs/{jd}.yaml.",
        "Raw per-night zen.*.sum.uvh5 in the working area are 0-byte "
        "placeholders; only calibrated stages are kept.",
        "ptsrc products exist in two generations; the "
        "'-original-wrong-spec*' variants are deprecated (spectral-index sign "
        "error, repaired via scripts/update_ptsrc_model.py over all 8640 "
        "chunks).",
    ],
    "products": products,
}

OUT_JSON.write_text(json.dumps(manifest, indent=2) + "\n")

csv_fields = ["product_id", "role", "path", "file_format", "n_files",
              "sky_components", "corruptions", "generator"]
with OUT_CSV.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
    w.writeheader()
    for p in products:
        w.writerow(p)

print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size} B) and {OUT_CSV}")
