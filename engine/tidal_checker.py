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
            modes = set(grp["mode"].astype(str).str.strip().str.lower())
            if len(modes) != 1:
                raise ValueError(
                    f"Machine {mid} punya lebih dari satu mode tidal: {sorted(modes)}"
                )

            e_mins = set(grp["E_min"].astype(float))
            if len(e_mins) != 1:
                raise ValueError(
                    f"Machine {mid} punya lebih dari satu E_min: {sorted(e_mins)}"
                )

            buffers = set(grp["buffer_time"].astype(float))
            if len(buffers) != 1:
                raise ValueError(
                    f"Machine {mid} punya lebih dari satu buffer_time: {sorted(buffers)}"
                )

            self._tidal_machine_ids.add(mid)
            self._constraints[mid] = {
                "port_name":   grp["port_name"].iloc[0],
                "mode":        grp["mode"].iloc[0],
                "E_min":       float(grp["E_min"].iloc[0]),
                "buffer_time": float(grp["buffer_time"].iloc[0]),
                "ships":       set(grp["ship_name"].tolist()),
            }

        # Load feasible windows
        # Mode `alur` memakai arrival/departure windows.
        # Mode `sandar` memakai raw feasible windows.
        self._arr_starts: dict[int, np.ndarray] = {}
        self._arr_ends:   dict[int, np.ndarray] = {}
        self._dep_starts: dict[int, np.ndarray] = {}
        self._dep_ends:   dict[int, np.ndarray] = {}
        self._raw_starts: dict[int, np.ndarray] = {}
        self._raw_ends:   dict[int, np.ndarray] = {}

        windows_df = pd.read_csv(windows_path)
        for machine_id, grp in windows_df.groupby("machine_id"):
            mid = int(machine_id)
            grp_sorted = grp.sort_values("raw_window_start").reset_index(drop=True)
            mode = self._constraints[mid]["mode"]

            self._raw_starts[mid] = grp_sorted["raw_window_start"].to_numpy(dtype=float)
            self._raw_ends[mid]   = grp_sorted["raw_window_end"].to_numpy(dtype=float)

            if mode == "alur":
                # Arrival intervals: [arrival_start, arrival_end]
                arr_mask = grp_sorted["arrival_start"] < grp_sorted["arrival_end"]
                arr = grp_sorted[arr_mask]
                self._arr_starts[mid] = arr["arrival_start"].to_numpy(dtype=float)
                self._arr_ends[mid]   = arr["arrival_end"].to_numpy(dtype=float)

                # Departure intervals: [departure_start, departure_end]
                dep_mask = grp_sorted["departure_start"] < grp_sorted["departure_end"]
                dep = grp_sorted[dep_mask]
                self._dep_starts[mid] = dep["departure_start"].to_numpy(dtype=float)
                self._dep_ends[mid]   = dep["departure_end"].to_numpy(dtype=float)
            else:
                self._arr_starts[mid] = np.array([], dtype=float)
                self._arr_ends[mid]   = np.array([], dtype=float)
                self._dep_starts[mid] = np.array([], dtype=float)
                self._dep_ends[mid]   = np.array([], dtype=float)

        # Hourly lookup (lazy)
        self._hourly_path = Path(hourly_path)
        self._hourly: dict[int, pd.DataFrame] | None = None

        n_arr = sum(len(v) for v in self._arr_starts.values())
        n_dep = sum(len(v) for v in self._dep_starts.values())
        n_raw = sum(len(v) for v in self._raw_starts.values())
        print(
            f"[TidalChecker] Loaded: "
            f"{len(self._tidal_machine_ids)} tidal machines | "
            f"{n_arr} arrival windows | {n_dep} departure windows | "
            f"{n_raw} raw windows"
        )

    # Public API

    @property
    def tidal_machine_ids(self) -> set[int]:
        return self._tidal_machine_ids

    def has_tidal_constraint(self, machine_id: int, ship_name: str | None = None) -> bool:
        mid = int(machine_id)
        if mid not in self._tidal_machine_ids:
            return False
        if ship_name is None:
            return True
        allowed_ships = self._constraints[mid]["ships"]
        return ship_name in allowed_ships

    def get_constraint(self, machine_id: int) -> dict | None:
        return self._constraints.get(int(machine_id))

    def is_feasible(
        self,
        machine_id: int,
        start_h: float,
        duration_h: float,
        ship_name: str | None = None,
    ) -> bool:
        mid = int(machine_id)
        if not self.has_tidal_constraint(mid, ship_name):
            return True

        mode = self._constraints[mid]["mode"]
        end_h = start_h + duration_h

        if mode == "alur":
            before_ok = self._point_in_intervals(
                self._arr_starts.get(mid, np.array([])),
                self._arr_ends.get(mid, np.array([])),
                start_h,
            )
            if not before_ok:
                return False

            after_ok = self._point_in_intervals(
                self._dep_starts.get(mid, np.array([])),
                self._dep_ends.get(mid, np.array([])),
                end_h,
            )
            return after_ok

        return self._interval_overlaps_raw_window(
            self._raw_starts.get(mid, np.array([])),
            self._raw_ends.get(mid, np.array([])),
            start_h,
            end_h,
        )

    def find_next_start(
        self,
        machine_id: int,
        earliest_h: float,
        duration_h: float,
        ship_name: str | None = None,
    ) -> float:
        mid = int(machine_id)
        if not self.has_tidal_constraint(mid, ship_name):
            return earliest_h

        mode = self._constraints[mid]["mode"]
        if mode == "sandar":
            return self._find_next_start_sandar(mid, earliest_h, duration_h)

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
        ship_name: str | None = None,
    ) -> float:
        if not self.has_tidal_constraint(machine_id, ship_name):
            return 0.0
        best_start = self.find_next_start(machine_id, earliest_h, duration_h, ship_name=ship_name)
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
                "mode":          cfg["mode"],
                "E_min":         cfg["E_min"],
                "buffer_time":   cfg["buffer_time"],
                "n_raw_windows": len(self._raw_starts.get(mid, [])),
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

    @staticmethod
    def _interval_overlaps_raw_window(
        raw_starts: np.ndarray,
        raw_ends: np.ndarray,
        start_h: float,
        end_h: float,
    ) -> bool:
        """
        Mode sandar:
        feasible jika interval [start_h, end_h] overlap dengan minimal satu
        raw feasible window [raw_start, raw_end].
        """
        if len(raw_starts) == 0 or end_h < start_h:
            return False

        idx = int(np.searchsorted(raw_ends, start_h, side="left"))
        if idx >= len(raw_starts):
            return False
        return raw_starts[idx] <= end_h

    def _find_next_start_sandar(
        self,
        machine_id: int,
        earliest_h: float,
        duration_h: float,
    ) -> float:
        """
        Cari start S >= earliest_h sehingga [S, S+p] overlap dengan minimal
        satu raw feasible window [w_start, w_end].

        Untuk satu raw window, himpunan S yang feasible adalah:
        S in [w_start - p, w_end]
        """
        raw_starts = self._raw_starts.get(machine_id, np.array([]))
        raw_ends   = self._raw_ends.get(machine_id,   np.array([]))

        if len(raw_starts) == 0:
            return float("inf")

        for w_start, w_end in zip(raw_starts, raw_ends):
            candidate = max(earliest_h, w_start - duration_h)
            if candidate <= w_end:
                return candidate

        return float("inf")

    def _ensure_hourly_loaded(self):
        if self._hourly is not None:
            return
        self._hourly = {}
        if not self._hourly_path.exists():
            return
        df = pd.read_csv(self._hourly_path)
        df["hour_offset"] = df["hour_offset"].round(0).astype(int)
        for machine_id, grp in df.groupby("machine_id"):
            self._hourly[int(machine_id)] = grp.sort_values("hour_offset").reset_index(drop=True)


# CLI Sanity Check

if __name__ == "__main__":
    checker = TidalChecker()

    print("\n=== Summary TidalChecker ===")
    print(checker.summary().to_string(index=False))

    print("\n=== Verifikasi Ringkas ===")
    print("Mode `alur`   : cek arrival window + departure window seperti sebelumnya")
    print("Mode `sandar` : feasible jika [S_lj, C_lj] overlap dengan minimal satu raw feasible window")
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
