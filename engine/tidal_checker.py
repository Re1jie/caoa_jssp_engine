import pandas as pd
import numpy as np
from pathlib import Path

# Konstanta

_BASE_DIR = Path(__file__).parent.parent
_CONSTRAINTS_PATH = _BASE_DIR / "data/processed/tidal_constraints.csv"
_WINDOWS_PATH     = _BASE_DIR / "data/processed/tidal_feasible_windows.csv"
_HOURLY_PATH      = _BASE_DIR / "data/processed/tidal_hourly_lookup.csv"


class TidalChecker:
    def __init__(
        self,
        constraints_path: str | Path = _CONSTRAINTS_PATH,
        windows_path:     str | Path = _WINDOWS_PATH,
        hourly_path:      str | Path = _HOURLY_PATH,
    ):
        # Load constraints
        self._constraints: dict[int, dict] = {}
        self._tidal_machine_ids: set[int] = set()

        constraints_df = pd.read_csv(constraints_path)
        for machine_id, grp in constraints_df.groupby("machine_id"):
            mid = int(machine_id)
            self._tidal_machine_ids.add(mid)
            self._constraints[mid] = {
                "port_name":   grp["port_name"].iloc[0],
                "E_min":       float(grp["E_min"].iloc[0]),
                "buffer_time": float(grp["buffer_time"].iloc[0]),
                "ships":       set(grp["ship_name"].tolist()),
            }

        # Load feasible windows
        # Dua set array terpisah per machine:
        #   _arr: arrival windows  → A_{l,j} harus masuk interval [arr_start, arr_end]
        #   _dep: departure windows → C_{l,j} harus masuk interval [dep_start, dep_end]
        self._arr_starts: dict[int, np.ndarray] = {}
        self._arr_ends:   dict[int, np.ndarray] = {}
        self._dep_starts: dict[int, np.ndarray] = {}
        self._dep_ends:   dict[int, np.ndarray] = {}

        windows_df = pd.read_csv(windows_path)
        for machine_id, grp in windows_df.groupby("machine_id"):
            mid = int(machine_id)
            grp_sorted = grp.sort_values("raw_window_start").reset_index(drop=True)

            # Arrival intervals: [arrival_start, arrival_end]
            # A valid arrival window has arrival_start < arrival_end
            arr_mask = grp_sorted["arrival_start"] < grp_sorted["arrival_end"]
            arr = grp_sorted[arr_mask]
            self._arr_starts[mid] = arr["arrival_start"].values
            self._arr_ends[mid]   = arr["arrival_end"].values

            # Departure intervals: [departure_start, departure_end]
            dep_mask = grp_sorted["departure_start"] < grp_sorted["departure_end"]
            dep = grp_sorted[dep_mask]
            self._dep_starts[mid] = dep["departure_start"].values
            self._dep_ends[mid]   = dep["departure_end"].values

        # Hourly lookup (lazy)
        self._hourly_path = Path(hourly_path)
        self._hourly: dict[int, pd.DataFrame] | None = None

        n_arr = sum(len(v) for v in self._arr_starts.values())
        n_dep = sum(len(v) for v in self._dep_starts.values())
        print(
            f"[TidalChecker] Loaded: "
            f"{len(self._tidal_machine_ids)} tidal machines | "
            f"{n_arr} arrival windows | {n_dep} departure windows"
        )

    # Public API

    @property
    def tidal_machine_ids(self) -> set[int]:
        return self._tidal_machine_ids

    def has_tidal_constraint(self, machine_id: int) -> bool:
        return int(machine_id) in self._tidal_machine_ids

    def get_constraint(self, machine_id: int) -> dict | None:
        return self._constraints.get(int(machine_id))

    def is_feasible(
        self,
        machine_id: int,
        start_h: float,
        duration_h: float,
    ) -> bool:
        mid = int(machine_id)
        if mid not in self._tidal_machine_ids:
            return True

        end_h = start_h + duration_h

        # Cek 1: A_{l,j} masuk arrival window?
        before_ok = self._point_in_intervals(
            self._arr_starts.get(mid, np.array([])),
            self._arr_ends.get(mid, np.array([])),
            start_h,
        )
        if not before_ok:
            return False

        # Cek 2: C_{l,j} masuk departure window?
        after_ok = self._point_in_intervals(
            self._dep_starts.get(mid, np.array([])),
            self._dep_ends.get(mid, np.array([])),
            end_h,
        )
        return after_ok

    def find_next_start(
        self,
        machine_id: int,
        earliest_h: float,
        duration_h: float,
    ) -> float:
        mid = int(machine_id)
        if mid not in self._tidal_machine_ids:
            return earliest_h

        arr_starts = self._arr_starts.get(mid, np.array([]))
        arr_ends   = self._arr_ends.get(mid,   np.array([]))
        dep_starts = self._dep_starts.get(mid, np.array([]))
        dep_ends   = self._dep_ends.get(mid,   np.array([]))

        if len(arr_starts) == 0 or len(dep_starts) == 0:
            return float("inf")

        best_A = float("inf")

        for i in range(len(arr_starts)):
            A_lo = max(arr_starts[i], earliest_h)
            A_hi = arr_ends[i]

            if A_lo > A_hi:
                continue  # arrival window seluruhnya sebelum earliest_h

            # C minimum untuk A_lo
            C_lo = A_lo + duration_h

            # Cari departure window j pertama yang bisa mengakomodasi C >= C_lo
            # C harus dalam [dep_start_j, dep_end_j]
            # ↔ A harus dalam [dep_start_j - p, dep_end_j - p]
            # dep_ends sorted → binary search untuk dep_end >= C_lo
            j_start = int(np.searchsorted(dep_ends, C_lo, side="left"))

            for j in range(j_start, len(dep_starts)):
                # Kombinasi constraint:
                # A ∈ [A_lo, A_hi] ∩ [dep_start_j - p, dep_end_j - p]
                A_lo_j = max(A_lo, dep_starts[j] - duration_h)
                A_hi_j = min(A_hi, dep_ends[j]   - duration_h)

                if A_lo_j > A_hi_j:
                    if dep_starts[j] - duration_h > A_hi:
                        break  # dep windows makin kanan, tidak akan overlap lagi
                    continue

                # Kombinasi valid → ambil A terkecil
                candidate = A_lo_j
                if candidate < best_A:
                    best_A = candidate

                # Karena dep windows sorted dan kita ambil A_lo_j = max(A_lo, ...),
                # window j berikutnya akan memberi A_lo_j lebih besar atau sama
                break  # dep window pertama yang valid sudah optimal untuk i ini

            # Early exit: jika found A = A_lo (tidak bisa lebih kecil untuk i)
            if best_A <= A_lo:
                break

        return best_A

    def delay_hours(
        self,
        machine_id: int,
        earliest_h: float,
        duration_h: float,
    ) -> float:
        if not self.has_tidal_constraint(machine_id):
            return 0.0
        best_start = self.find_next_start(machine_id, earliest_h, duration_h)
        if best_start == float("inf"):
            return float("inf")
        return max(0.0, best_start - earliest_h)

    def get_elevation_at(self, machine_id: int, hour_offset: float) -> float | None:
        """Kembalikan tidal elevation pada jam tertentu (lazy load hourly data)."""
        self._ensure_hourly_loaded()
        mid = int(machine_id)
        if mid not in self._hourly:
            return None
        df = self._hourly[mid]
        h_floor = int(hour_offset)
        row = df[df["hour_offset"] == h_floor]
        if row.empty:
            return None
        return float(row["tidal_elevation"].iloc[0])

    def summary(self) -> pd.DataFrame:
        """Ringkasan constraint per machine untuk debugging."""
        rows = []
        for mid, cfg in self._constraints.items():
            rows.append({
                "machine_id":    mid,
                "port_name":     cfg["port_name"],
                "E_min":         cfg["E_min"],
                "buffer_time":   cfg["buffer_time"],
                "n_arr_windows": len(self._arr_starts.get(mid, [])),
                "n_dep_windows": len(self._dep_starts.get(mid, [])),
                "ships":         ", ".join(sorted(cfg["ships"])),
            })
        return pd.DataFrame(rows).sort_values("machine_id").reset_index(drop=True)

    # Private Helpers

    @staticmethod
    def _point_in_intervals(
        starts: np.ndarray,
        ends: np.ndarray,
        t: float,
    ) -> bool:
        """Binary search: apakah titik t berada dalam salah satu [starts[i], ends[i]]?"""
        if len(starts) == 0:
            return False
        idx = int(np.searchsorted(starts, t, side="right")) - 1
        if idx >= 0 and starts[idx] <= t <= ends[idx]:
            return True
        return False

    def _ensure_hourly_loaded(self):
        if self._hourly is not None:
            return
        self._hourly = {}
        if not self._hourly_path.exists():
            return
        df = pd.read_csv(self._hourly_path)
        df["hour_offset"] = df["hour_offset"].round(0).astype(int)
        for machine_id, grp in df.groupby("machine_id"):
            self._hourly[int(machine_id)] = grp.reset_index(drop=True)


# CLI Sanity Check

if __name__ == "__main__":
    checker = TidalChecker()

    print("\n=== Summary TidalChecker ===")
    print(checker.summary().to_string(index=False))

    print("\n=== Verifikasi Contoh User ===")
    print("Skenario: kapal sandar jam 8-10 (A=8, p=2, C=10), buffer=2")
    print("  Cek sebelum : [6, 8]  harus dalam suatu raw window")
    print("  Cek sesudah : [10, 12] harus dalam suatu raw window")
    print("  Tide jam 8-10 (selama sandar) → TIDAK dipedulikan")
    print()

    print("=== Contoh Query ===")
    test_cases = [
        # (machine_id, start_h, duration_h, keterangan)
        (6,  206.0, 10.0, "KUMAI — cek arrival+departure independen"),
        (70, 500.0,  5.0, "AGATS — arrival jam 500, departure jam 505"),
        (59, 100.0, 19.0, "SAMPIT — durasi 19j, bisa pakai window terpisah"),
        (71, 400.0,  2.0, "MERAUKE E_min=4.0m — ketat"),
        (0,  100.0,  5.0, "KENDARI — tidak ada constraint"),
    ]
    for mid, sh, dur, desc in test_cases:
        feasible   = checker.is_feasible(mid, sh, dur)
        delay      = checker.delay_hours(mid, sh, dur)
        next_start = checker.find_next_start(mid, sh, dur)
        status = "✅ OK" if feasible else f"⏳ DELAY {delay:.1f}h (next start: {next_start:.1f}h)"
        print(f"  [{mid:>3}] {desc}")
        print(f"          → {status}")
