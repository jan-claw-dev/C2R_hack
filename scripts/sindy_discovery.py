import json
from pathlib import Path

import numpy as np
import pysindy as ps

DATA = {
    "Koch": "Koch_model_processed.npy",
    "HighFidelity": "high_fidelity_sim_processed.npy",
}

SENSOR_INDICES = [0, 25, 50, 75, 99]
DT = 0.1

OUTPUT_JSON = "reports/sindy_results.json"
OUTPUT_MD = "reports/sindy_results.md"


def fit_sindy(name: str, data: np.ndarray) -> dict:
    X = data[:, SENSOR_INDICES]
    dXdt = np.gradient(X, axis=0) / DT

    feature_names = [f"u{i}" for i in SENSOR_INDICES]
    optimizer = ps.STLSQ(threshold=0.001)
    library = ps.PolynomialLibrary(degree=3)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.feature_names = feature_names
    model.fit(X, t=DT, x_dot=dXdt)

    predictions = model.predict(X)
    mse = float(np.mean((predictions - dXdt) ** 2))
    ss_res = np.sum((predictions - dXdt) ** 2)
    ss_tot = np.sum((dXdt - np.mean(dXdt, axis=0)) ** 2)
    score = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    eqns = model.equations()

    return {
        "name": name,
        "equations": eqns,
        "mse_derivative": mse,
        "score": score,
        "num_terms": sum(len(eq.split("+")) for eq in eqns),
    }


def main() -> None:
    results = []
    for label, filename in DATA.items():
        data = np.load(Path(filename))
        results.append(fit_sindy(label, data))

    koch = np.load(Path(DATA["Koch"]))
    real = np.load(Path(DATA["HighFidelity"]))
    diff = real - koch
    results.append(fit_sindy("Residual", diff))

    Path("reports").mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w") as f_json:
        json.dump(results, f_json, indent=2)

    with open(OUTPUT_MD, "w") as f_md:
        f_md.write("# SINDy Discovery Report\n\n")
        f_md.write(f"Sensor indices: {SENSOR_INDICES}\n\n")
        for entry in results:
            f_md.write(f"## {entry['name']}\n")
            f_md.write(f"- Derivative MSE: {entry['mse_derivative']:.6e}\n")
            f_md.write(f"- Score (R²-like): {entry['score']:.4f}\n")
            f_md.write(f"- Number of active terms: {entry['num_terms']}\n")
            f_md.write("- Equations to describe dx/dt:\n")
            for eq in entry["equations"]:
                f_md.write(f"  - `{eq}`\n")
            f_md.write("\n")

    print(f"SINDy discovery completed. Results -> {OUTPUT_MD}")


def entry_point() -> None:
    main()


if __name__ == "__main__":
    entry_point()
