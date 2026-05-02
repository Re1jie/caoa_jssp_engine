from pathlib import Path

import numpy as np


def _latest_nonparametric_dir(root="data/results/nonparametric"):
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = [path for path in root_path.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _load_nonparametric_curves(output_dir):
    summary_dir = output_dir / "convergence_summary"
    curves = {}
    for mean_path in sorted(summary_dir.glob("*_mean_convergence.npy")):
        label = mean_path.name.replace("_mean_convergence.npy", "").upper()
        std_path = summary_dir / f"{label.lower()}_std_convergence.npy"
        curves[label] = {
            "mean": np.load(mean_path),
            "std": np.load(std_path) if std_path.exists() else None,
        }
    return curves


def _load_legacy_curves():
    results = {
        "CAOA": "data/results/caoa/caoa_convergence_curve.npy",
        "CAOASSR": "data/results/caoassr/caoassr_convergence_curve.npy",
        "GWO": "data/results/gwo/gwo_convergence_curve.npy",
    }
    curves = {}
    for label, path_str in results.items():
        path = Path(path_str)
        if path.exists():
            curves[label] = {"mean": np.load(path), "std": None}
    return curves


def plot_comparison(output_dir=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib tidak tersedia, plot tidak dibuat ({exc})")
        return

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
    else:
        output_path = _latest_nonparametric_dir()

    if output_path is not None:
        curves = _load_nonparametric_curves(output_path)
        figure_path = output_path / "convergence_mean_comparison.png"
    else:
        curves = {}
        figure_path = Path("data/results/convergence_comparison.png")

    if not curves:
        curves = _load_legacy_curves()
        figure_path = Path("data/results/convergence_comparison.png")

    if not curves:
        print("Warning: tidak ada file kurva konvergensi yang ditemukan")
        return

    plt.figure(figsize=(10, 6))

    for label, data in curves.items():
        mean_curve = data["mean"]
        std_curve = data["std"]
        x = np.arange(1, len(mean_curve) + 1)
        plt.plot(x, mean_curve, label=label, linewidth=2)
        if std_curve is not None:
            plt.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
                linewidth=0,
            )
        print(f"Loaded {label}: final mean value = {mean_curve[-1]:.2f}")

    plt.title("Perbandingan Konvergensi Algoritma (30 Run)")
    plt.xlabel("Iterasi")
    plt.ylabel("Total Tardiness (Jam)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150)
    print(f"\nGrafik perbandingan disimpan di: {figure_path}")


if __name__ == "__main__":
    plot_comparison()
