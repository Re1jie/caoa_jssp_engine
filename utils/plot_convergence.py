import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison():
    results = {
        "CAOA": "data/results/caoa/caoa_convergence_curve.npy",
        "CAOASSR": "data/results/caoassr/caoassr_convergence_curve.npy",
        "GWO": "data/results/gwo/gwo_convergence_curve.npy"
    }

    plt.figure(figsize=(10, 6))
    
    for label, path_str in results.items():
        path = Path(path_str)
        if path.exists():
            data = np.load(path)
            plt.plot(data, label=label, linewidth=2)
            print(f"Loaded {label}: final value = {data[-1]:.2f}")
        else:
            print(f"Warning: {path_str} tidak ditemukan")

    plt.title("Perbandingan Konvergensi Algoritma (Minimasi Total Tardiness)")
    plt.xlabel("Iterasi")
    plt.ylabel("Total Tardiness (Jam)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = "data/results/convergence_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nGrafik perbandingan disimpan di: {output_path}")

if __name__ == "__main__":
    plot_comparison()
