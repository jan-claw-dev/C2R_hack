# RDE-DA-SHRED Submission Bundle

This project pairs a flexible DA-SHRED stack (`Cheap2Rich.py`) with discovery scripts that probe latent and multi-scale structure in the rotated detonation engine (RDE) datasets. The repository contains the Koch-based low-fidelity simulations, the high-fidelity measurements, the LF/HF fusion code, and the auxiliary discovery scripts described below.

## Architecture overview

- **LF Pathway:** Each sensor history is passed through a shallow LSTM (with optional temporal convolutions) and normalized to produce a latent representation, then decoded with a residual MLP.
- **HF Pathway:** Sensor residuals are modeled by a DA-SHRED module that learns a high-frequency correction via spatial deformation, gating, and optional frequency-domain attention.
- **Latent alignment:** Various experiments replace the old GAN with contrastive projection, Fourier-gated outputs, or a latent-space SINDy/Lasso aligner before decoding.

## Latest benchmark (latent SINDy aligner run)

| Branch | LF+HF RMSE | Full SSIM (LF+HF) | Notes |
| --- | --- | --- | --- |
| `main` | 0.1067 | 0.3574 | Baseline pipeline with GAN aligner + HF gating and frequency dropout. |
| `contrastive-latent` | 0.1034 | 0.3657 | Replaces the GAN with a contrastive projector; very similar reconstruction but simpler training. |
| `sindy-latent` | 0.1552 | 0.2073 | Fits a polynomial/Lasso aligner in the latent space prior to decoding—still improving RMSE though SSIM is lower while we tune convergence. |

*All runs use the provided `RDE_four_panel_comparison.png` and `sparse_freq_RDE_results.png` plots in the root directory for visual comparison.*

## Branch-specific notes

- `main`: Uses the GAN-based aligner plus the HF gating/frequency attention experiments described above. The run artifacts live in `RDE_*` PNGs.
- `contrastive-latent`: Aligns LF latents with a contrastive projector and keeps the rest of the pipeline identical; see the same plots for comparison and this branch’s PR for implementation details.
- `frequency-attn`: Adds a Fourier-domain gating layer that learns per-frequency multipliers before injecting HF corrections.
- `sindy-latent`: Fits either SINDy or a Lasso+Polynomial aligner on the first 80 latent sample steps and uses the predicted delta to shift the decoded latents; the modified README and plots highlight the resulting RMSE/SSIM.
- `sindy-discovery`: Contains the PySINDy discovery script (`scripts/sindy_discovery.py`) plus a `reports/sindy_*` summary of the identified latent equations.
- `mamba-discovery`: Builds multi-scale Ricker features, fits a ridge model to the latent derivative, and saves the derivative/forecast plots under `reports/mamba_*`.

## Run artifacts

- `RDE_four_panel_comparison.png` / `RDE_four_panel_validation.png`: Side-by-side animation of simulation, real data, LF-only, and LF+HF reconstructions.
- `sparse_freq_RDE_results.png`: HF reconstruction + frequency spectrum.
- Discovery plot files (`reports/sindy_*.png`, `reports/mamba_*.png`) illustrate derivative predictions for Koch/high-fidelity/residual fields.

## Discovery workflows

- **SINDy discovery** (`scripts/sindy_discovery.py`): Applies PySINDy to the Koch, high-fidelity, and residual datasets using five chosen sensor indices. Results appear in `reports/sindy_results.[md|json]`.
- **MAMBA-style discovery** (`scripts/mamba_discovery.py`): Constructs time-frequency features with Ricker kernels and fits a ridge model; reports go to `reports/mamba_results.[md|json]`.

## Getting started

1. Install dependencies (the repo bundles a `venv` for reference): `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
2. Run `python Cheap2Rich.py` (or inspect `baselines_for_results.ipynb`) to reproduce the LF/HF training + reconstructions.
3. Rerun the discovery scripts (and push their reports) if you change `SENSOR_INDICES`, thresholds, or aligner configurations.
