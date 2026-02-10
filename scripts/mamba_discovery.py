import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

DATA = {
    "Koch": "Koch_model_processed.npy",
    "HighFidelity": "high_fidelity_sim_processed.npy",
}

SENSOR_INDICES = [0, 25, 50, 75, 99]
DT = 0.1
SCALES = [2, 4, 8, 16]
OUTPUT_JSON = "reports/mamba_results.json"
OUTPUT_MD = "reports/mamba_results.md"


def ricker_wavelet(length: int, width: float) -> np.ndarray:
    x = np.linspace(-length / 2, length / 2, length)
    return (1 - (x ** 2) / (width ** 2)) * np.exp(-x ** 2 / (2 * width ** 2))


def multi_scale_features(X: np.ndarray, include_values: bool) -> tuple[np.ndarray, list[str]]:
    features = []
    names = []
    if include_values:
        features.append(X)
        names.extend([f"u{i}" for i in SENSOR_INDICES])

    for local_idx, sensor_idx in enumerate(SENSOR_INDICES):
        conv_feats = []
        series = X[:, local_idx]
        for width in SCALES:
            kernel = ricker_wavelet(int(width * 6 + 1), width)
            conv_feats.append(np.convolve(series, kernel, mode="same"))
        cwt_mat = np.vstack(conv_feats).T
        features.append(cwt_mat)
        names.extend([f"x{sensor_idx}_cwt_w{width}" for width in SCALES])

    return np.concatenate(features, axis=1), names


def plot_mamba(name: str, actual: np.ndarray, predicted: np.ndarray) -> None:
    idx = SENSOR_INDICES[0]
    t = np.arange(actual.shape[0]) * DT
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, actual[:, idx], label="True dx/dt")
    axes[0].plot(t, predicted[:, idx], label="MAMBA predict", linestyle="--")
    axes[0].set_ylabel("Derivative")
    axes[0].legend()

    axes[1].plot(t, actual[:, idx] - predicted[:, idx], color="tab:purple")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Time")

    Path("reports").mkdir(exist_ok=True)
    plot_path = Path("reports") / f"mamba_{name.replace(' ', '_')}.png"
    fig.suptitle(f"{name} derivative comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def run_mamba(name: str, data: np.ndarray, include_values: bool) -> dict:
    X = data[:, SENSOR_INDICES]
    dXdt = np.gradient(X, axis=0) / DT

    features, names = multi_scale_features(X, include_values)
    model = Ridge(alpha=1.0)
    model.fit(features, dXdt)

    predictions = model.predict(features)
    mse = float(np.mean((predictions - dXdt) ** 2))
    ss_res = np.sum((predictions - dXdt) ** 2)
    ss_tot = np.sum((dXdt - np.mean(dXdt, axis=0)) ** 2)
    score = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    coef_norm = np.linalg.norm(model.coef_, axis=0)
    top_feats = [names[i] for i in np.argsort(-coef_norm)[:5]]

    tag = f"{name}_{'values' if include_values else 'cwt'}"
    plot_mamba(tag, dXdt, predictions)

    return {
        "name": name,
        "mode": "cwt+values" if include_values else "cwt-only",
        "mse": mse,
        "score": score,
        "top_features": top_feats,
        "num_features": features.shape[1],
    }


def main() -> None:
    results = []
    Path("reports").mkdir(exist_ok=True)

    for label, filename in DATA.items():
        arr = np.load(Path(filename))
        results.append(run_mamba(label, arr, include_values=True))
        results.append(run_mamba(label, arr, include_values=False))

    with open(Path(OUTPUT_JSON), "w") as outfile:
        json.dump(results, outfile, indent=2)

    with open(Path(OUTPUT_MD), "w") as md:
        md.write("# MAMBA-inspired Multi-scale Discovery\n\n")
        md.write(f"Sensor indices: {SENSOR_INDICES}\n\n")
        for entry in results:
            md.write(f"## {entry['name']} ({entry['mode']})\n")
            md.write(f"- Feature count: {entry['num_features']}\n")
            md.write(f"- Derivative MSE: {entry['mse']:.6e}\n")
            md.write(f"- Score: {entry['score']:.4f}\n")
            md.write(f"- Top features: {entry['top_features']}\n")
            md.write(f"![{entry['name']} {entry['mode']} derivative](mamba_{entry['name']}_{'values' if entry['mode']== 'cwt+values' else 'cwt-only'}.png)\n\n")

    print(f"MAMBA discovery finished. Report at {OUTPUT_MD}")


if __name__ == "__main__":
    main()
