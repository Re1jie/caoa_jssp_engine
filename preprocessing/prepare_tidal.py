import pandas as pd
import numpy as np
from pathlib import Path

# ─── Konstanta ────────────────────────────────────────────────────────────────

T_ZERO = pd.Timestamp("2025-01-01 00:00:00")

RAW_TIDAL_DATA   = Path("data/raw/tidal_data.csv")
RAW_TIDAL_RULES  = Path("data/raw/tidal_rules.csv")
MACHINE_MASTER   = Path("data/processed/machine_master.csv")
OUT_DIR          = Path("data/processed")

# ─── Helper Functions ─────────────────────────────────────────────────────────

def parse_tidal_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse timestamp dari tidal_data.csv.
    Handle kasus jam '24:00:00' yang tidak standar (= 00:00:00 hari berikutnya).
    Contoh: '2025-01-01 24:00:00' → '2025-01-02 00:00:00'
    """
    # Pisahkan date dan time
    parts = series.str.split(" ", n=1, expand=True)
    dates = parts[0]
    times = parts[1]

    # Identifikasi baris dengan jam 24
    mask_24 = times.str.startswith("24:")

    # Ganti '24:xx:xx' → '00:xx:xx' dan tambahkan 1 hari ke tanggalnya
    times_fixed = times.where(~mask_24, times.str.replace("^24:", "00:", regex=True))
    dates_fixed = dates.copy()
    dates_fixed[mask_24] = (
        pd.to_datetime(dates[mask_24]) + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    combined = dates_fixed + " " + times_fixed
    return pd.to_datetime(combined, format="%Y-%m-%d %H:%M:%S")


def extract_contiguous_windows(
    hour_offsets: np.ndarray,
    is_feasible: np.ndarray,
) -> list[tuple[float, float]]:
    windows = []
    in_window = False
    start = None

    for i, (h, feasible) in enumerate(zip(hour_offsets, is_feasible)):
        if feasible and not in_window:
            in_window = True
            start = h
        elif not feasible and in_window:
            in_window = False
            # +1: jam terakhir feasible mencakup interval [H, H+1)
            windows.append((start, hour_offsets[i - 1] + 1.0))

    # Tutup window yang masih terbuka di akhir data
    if in_window:
        windows.append((start, hour_offsets[-1] + 1.0))

    return windows


# ─── Step 1: Load & Join Tidal Rules dengan Machine Master ───────────────────

print("📂 Loading raw data...")
tidal_rules  = pd.read_csv(RAW_TIDAL_RULES)
machine_master = pd.read_csv(MACHINE_MASTER)

if "mode" not in tidal_rules.columns:
    tidal_rules["mode"] = np.where(tidal_rules["buffer_time"] > 0, "alur", "sandar")

tidal_rules["mode"] = tidal_rules["mode"].astype(str).str.strip().str.lower()
valid_modes = {"alur", "sandar"}
invalid_modes = sorted(set(tidal_rules["mode"]) - valid_modes)
if invalid_modes:
    raise ValueError(
        f"Mode tidal tidak dikenali: {invalid_modes}. "
        f"Mode valid hanya {sorted(valid_modes)}."
    )

# Join: tidal_rules.port_name ↔ machine_master.PELABUHAN_LOGIS
constraints = tidal_rules.merge(
    machine_master[["PELABUHAN_LOGIS", "machine_id"]],
    left_on="port_name",
    right_on="PELABUHAN_LOGIS",
    how="left",
)

# Periksa apakah ada port di tidal_rules yang tidak ada di machine_master
unmatched = constraints[constraints["machine_id"].isna()]["port_name"].unique()
if len(unmatched) > 0:
    print(f"  ⚠️  Port tidak ditemukan di machine_master: {unmatched}")

constraints = constraints.dropna(subset=["machine_id"])
constraints["machine_id"] = constraints["machine_id"].astype(int)

# Fallback defensif agar mode selalu konsisten dengan aturan bisnis terbaru.
inferred_mode = np.where(constraints["buffer_time"] > 0, "alur", "sandar")
mismatch_mode = constraints["mode"] != inferred_mode
if mismatch_mode.any():
    bad_rows = constraints.loc[mismatch_mode, ["port_name", "ship_name", "buffer_time", "mode"]]
    raise ValueError(
        "Ditemukan mode yang tidak konsisten dengan buffer_time.\n"
        "Aturan yang dipakai: buffer_time > 0 => alur, buffer_time = 0 => sandar.\n"
        f"Baris bermasalah:\n{bad_rows.to_string(index=False)}"
    )

# Simpan tidal_constraints.csv
constraints_out = constraints[[
    "machine_id", "port_name", "E_min", "buffer_time", "ship_name", "mode"
]].sort_values(["machine_id", "ship_name"]).reset_index(drop=True)

constraints_out.to_csv(OUT_DIR / "tidal_constraints.csv", index=False)
print(f"  ✅ Saved: tidal_constraints.csv ({len(constraints_out)} rows)")
print(constraints_out.to_string(index=False))

# ─── Step 2: Load Tidal Data & Hitung Hour Offset ────────────────────────────

print("\n📂 Loading tidal_data.csv (~148K rows)...")
tidal_raw = pd.read_csv(RAW_TIDAL_DATA, dtype={"port_name": str, "tidal_elevation": float})

tidal_raw["timestamp"] = parse_tidal_timestamps(tidal_raw["timestamp"])

# Hitung hour_offset dari T_ZERO (dalam satuan jam, konsisten dengan A_lj)
tidal_raw["hour_offset"] = (tidal_raw["timestamp"] - T_ZERO).dt.total_seconds() / 3600

print(f"  ✅ Parsed {len(tidal_raw):,} baris | Range: "
      f"{tidal_raw['timestamp'].min()} → {tidal_raw['timestamp'].max()}")

# ─── Step 3: Bangun Per-Machine Hourly Lookup & Feasible Windows ─────────────

# Ambil daftar unik konfigurasi tidal per port.
# `mode` menentukan interpretasi hard-constraint:
#   - alur   : arrival/departure window seperti logika existing
#   - sandar : selama durasi sandar harus overlap dengan minimal satu raw feasible window
port_constraints = (
    constraints[["port_name", "machine_id", "E_min", "buffer_time", "mode"]]
    .drop_duplicates()
    .sort_values(["machine_id", "port_name", "E_min", "buffer_time", "mode"])
    .reset_index(drop=True)
)

hourly_records  = []
window_records  = []

print("\n⚙️  Computing feasible windows per port...")

for _, cfg in port_constraints.iterrows():
    port_name   = cfg["port_name"]
    machine_id  = int(cfg["machine_id"])
    e_min       = float(cfg["E_min"])
    buffer_time = float(cfg["buffer_time"])
    mode        = cfg["mode"]

    # Filter tidal data untuk port ini, sort by time
    port_tidal = (
        tidal_raw[tidal_raw["port_name"] == port_name]
        .sort_values("hour_offset")
        .reset_index(drop=True)
    )

    if port_tidal.empty:
        print(f"  ⚠️  Tidak ada tidal data untuk port: {port_name}")
        continue

    hour_offsets = port_tidal["hour_offset"].values
    elevations   = port_tidal["tidal_elevation"].values
    is_feasible  = elevations >= e_min

    # ── Hourly lookup records ──
    for h, elev, feas in zip(hour_offsets, elevations, is_feasible):
        hourly_records.append({
            "machine_id":       machine_id,
            "port_name":        port_name,
            "mode":             mode,
            "E_min":            e_min,
            "hour_offset":      round(h, 4),
            "tidal_elevation":  round(elev, 2),
            "is_feasible":      feas,
        })

    raw_windows = extract_contiguous_windows(hour_offsets, is_feasible)

    for (w_start, w_end) in raw_windows:
        raw_width = w_end - w_start

        if mode == "alur":
            arrival_start   = w_start + buffer_time
            arrival_end     = w_end
            departure_start = w_start
            departure_end   = w_end - buffer_time

            # Simpan jika window cukup lebar untuk setidaknya satu peran
            # (arrival atau departure) — syarat: raw_width >= buffer_time
            if raw_width < buffer_time:
                continue
        else:
            # Mode sandar:
            # Selama interval sandar [S, C], cukup ada SATU titik/jam yang
            # tidal_elevation >= E_min. Karena itu preprocessing cukup
            # menyimpan raw feasible windows; checker nantinya akan mengecek
            # overlap [S, C] dengan salah satu raw window tersebut.
            arrival_start = np.nan
            arrival_end = np.nan
            departure_start = np.nan
            departure_end = np.nan

        window_records.append({
            "machine_id":       machine_id,
            "port_name":        port_name,
            "mode":             mode,
            "E_min":            e_min,
            "buffer_time":      buffer_time,
            "raw_window_start": round(w_start, 4),
            "raw_window_end":   round(w_end,   4),
            # Untuk mode `alur`, field ini dipakai langsung sebagai tidal window
            # arrival/departure. Untuk mode `sandar`, nilainya kosong karena
            # pengecekan dilakukan via overlap interval sandar dengan raw window.
            "arrival_start":    round(arrival_start,   4) if pd.notna(arrival_start) else np.nan,
            "arrival_end":      round(arrival_end,     4) if pd.notna(arrival_end) else np.nan,
            "departure_start":  round(departure_start, 4) if pd.notna(departure_start) else np.nan,
            "departure_end":    round(departure_end,   4) if pd.notna(departure_end) else np.nan,
        })

    n_raw     = len(raw_windows)
    if mode == "alur":
        n_usable = sum(1 for ws, we in raw_windows if (we - ws) >= buffer_time)
        note = f"{n_usable} usable (raw_width >= buffer)"
    else:
        n_usable = n_raw
        note = "semua raw window dipakai untuk overlap saat sandar"

    print(
        f"  [{machine_id:>3}] {port_name:<12} mode={mode:<6} "
        f"E_min={e_min:.1f}m buf={buffer_time}h "
        f"→ {n_raw} raw windows, {note}"
    )

# ─── Step 4: Simpan Output Files ─────────────────────────────────────────────

print("\n💾 Saving processed files...")

# tidal_hourly_lookup.csv
hourly_df = pd.DataFrame(hourly_records)
hourly_df = hourly_df.sort_values(["machine_id", "mode", "hour_offset"]).reset_index(drop=True)
hourly_df.to_csv(OUT_DIR / "tidal_hourly_lookup.csv", index=False)
print(f"  ✅ Saved: tidal_hourly_lookup.csv ({len(hourly_df):,} rows)")

# tidal_feasible_windows.csv
windows_df = pd.DataFrame(window_records)
windows_df = windows_df.sort_values(["machine_id", "mode", "raw_window_start"]).reset_index(drop=True)
windows_df.to_csv(OUT_DIR / "tidal_feasible_windows.csv", index=False)
print(f"  ✅ Saved: tidal_feasible_windows.csv ({len(windows_df):,} rows)")

# ─── Step 5: Sanity Check Summary ────────────────────────────────────────────

print("\n📊 Sanity Check — Windows per Machine:")
summary = (
    windows_df.groupby(["machine_id", "port_name", "mode", "E_min", "buffer_time"])
    .agg(
        n_windows    =("raw_window_start", "count"),
        min_raw_width=("raw_window_end", lambda x: round(
            (windows_df.loc[x.index, "raw_window_end"]
             - windows_df.loc[x.index, "raw_window_start"]).min(), 1)),
        max_raw_width=("raw_window_end", lambda x: round(
            (windows_df.loc[x.index, "raw_window_end"]
             - windows_df.loc[x.index, "raw_window_start"]).max(), 1)),
        min_arr_range=("arrival_end", lambda x: round(
            (windows_df.loc[x.index, "arrival_end"]
             - windows_df.loc[x.index, "arrival_start"]).min(), 1)),
        min_dep_range=("departure_end", lambda x: round(
            (windows_df.loc[x.index, "departure_end"]
             - windows_df.loc[x.index, "departure_start"]).min(), 1)),
    )
    .reset_index()
)
print(summary.to_string(index=False))
print("  (mode=alur: arr/dep range terisi | mode=sandar: arr/dep range kosong)")

print("\n✅ Tidal data preparation selesai!")
print(f"   Output files di: {OUT_DIR.resolve()}")
print(f"   T_ZERO (epoch): {T_ZERO}  ← konsisten dengan data_transformer.py")
print()
print("   LOGIKA `alur`  : before-buffer dan after-buffer bisa dari raw window BERBEDA")
print("   LOGIKA `sandar`: interval [S_lj, C_lj] harus overlap dengan minimal satu raw feasible window")
