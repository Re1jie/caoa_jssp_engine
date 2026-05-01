# CAOA JSSP Engine

Project ini menyelesaikan Job Shop Scheduling Problem untuk penjadwalan voyage. Setiap voyage punya beberapa operasi berurutan, tiap operasi harus masuk ke machine tertentu, dan schedule harus mematuhi:

- urutan operasi dalam voyage,
- waktu kedatangan awal,
- sailing time antar operasi,
- kapasitas berth per machine,
- constraint pasang surut,
- target selesai voyage.

Objective utama yang diminimasi adalah `total_tardiness`, yaitu total keterlambatan semua voyage terhadap due time masing-masing.

## Cara Membaca Project Ini

Alur berpikirnya sederhana:

1. Data operasi sudah disiapkan di `data/processed/`.
2. Baseline FCFS dibuat dulu sebagai pembanding.
3. Optimizer tidak langsung membuat schedule, tapi membuat vektor prioritas operasi.
4. Decoder mengubah vektor prioritas itu menjadi schedule nyata.
5. Schedule dicek feasibility-nya.
6. Metrics, timetable, debug report, dan convergence curve disimpan sebagai artefak eksperimen.

Bagian yang paling penting untuk dipahami adalah pemisahan antara optimizer dan decoder. Optimizer hanya mencari urutan prioritas yang bagus. Decoder yang memastikan prioritas itu menjadi schedule yang valid terhadap constraint dunia nyata.

## Entry Point

Ada tiga runner utama. Semuanya memakai data, decoder, metrics, dan baseline FCFS yang sama. Bedanya hanya optimizer yang dipakai.

```bash
python run_insertion_caoa.py
python run_insertion_caoassr.py
python run_insertion_gwo.py
```

Gunakan:

- `run_insertion_caoa.py` untuk eksperimen CAOA standar.
- `run_insertion_caoassr.py` untuk CAOA dengan SSR/RDK guidance.
- `run_insertion_gwo.py` untuk pembanding Grey Wolf Optimizer.

Visualisasi perbandingan konvergensi:

```bash
python utils/plot_convergence.py
```

Output plot disimpan ke:

```text
data/results/convergence_comparison.png
```

## Komponen Utama

- `engine/decoder_insertion.py`: decoder aktif berbasis insertion/capacity-aware slot search.
- `engine/fcfs.py`: baseline FCFS yang tetap menghormati precedence, sailing time, capacity, dan tidal constraint.
- `engine/tidal_checker.py`: pengecekan constraint pasang surut per machine dan ship.
- `engine/metrics.py`: perhitungan `total_tardiness`, `max_tardiness`, dan feasibility metrics.
- `engine/caoa.py`: optimizer CAOA.
- `engine/caoassr.py`: varian CAOA dengan SSR/RDK guidance dan diagnostic logging.
- `engine/gwo.py`: optimizer Grey Wolf Optimizer sebagai pembanding.
- `utils/data_loader.py`: loader data runtime dari `data/processed/`.

Jika ingin memahami runtime dari awal, urutan baca yang paling enak:

1. `run_insertion_caoassr.py`
2. `utils/data_loader.py`
3. `engine/decoder_insertion.py`
4. `engine/tidal_checker.py`
5. `engine/metrics.py`
6. `engine/caoassr.py`

## Input Data

File runtime utama:

```text
data/processed/jssp_data_sliced.csv
data/processed/job_target_time_sliced.csv
data/processed/machine_master.csv
data/processed/tidal_constraints.csv
data/processed/tidal_feasible_windows.csv
data/processed/tidal_hourly_lookup.csv
```

Kolom penting:

- Operasi: `job_id`, `voyage`, `op_seq`, `machine_id`, `A_lj`, `p_lj`, `TSail_lj`, opsional `ship_name`.
- Machine: `machine_id`, `num_berth`.
- Target: `job_id`, `voyage`, `T_j`, opsional `w_j`, `ship_name`.

`load_real_jssp_data(...)` memvalidasi bahwa pasangan `(job_id, voyage)` di data operasi dan target harus identik.

Makna kolom yang sering muncul:

- `A_lj`: waktu release atau arrival operasi.
- `p_lj`: durasi proses operasi.
- `TSail_lj`: sailing time setelah operasi selesai menuju operasi berikutnya.
- `S_lj`: waktu mulai operasi di schedule hasil decoder.
- `C_lj`: waktu selesai operasi di schedule hasil decoder.
- `T_j`: due window voyage, dihitung relatif dari arrival pertama voyage.

## Decoder dan Feasibility

`ActiveScheduleDecoder` menerima vektor kontinu `X` berdimensi jumlah operasi. Setiap nilai `X[i]` menjadi prioritas untuk satu operasi di `L_ref`.

Contoh mental model:

```text
X = [0.23, 0.91, 0.10, ...]
```

Nilai lebih kecil berarti operasi lebih diprioritaskan ketika operasi tersebut eligible. Tetapi operasi hanya boleh dipilih jika precedence voyage-nya sudah terpenuhi. Jadi optimizer bisa memberi prioritas kecil pada operasi ke-3, tetapi decoder tetap tidak akan menjadwalkannya sebelum operasi ke-1 dan ke-2 selesai.

Aturan scheduling:

- hanya operasi berikutnya dari setiap voyage yang eligible,
- operasi pertama release pada `A_lj`,
- operasi berikutnya release pada `C_lj(prev) + TSail_lj(prev)`,
- kandidat dipilih berdasarkan prioritas terkecil,
- start time dicari dari release time dan akhir interval yang sudah terjadwal pada machine,
- kapasitas dicek dengan overlap counting terhadap `num_berth`,
- tidal constraint diterapkan sebelum slot dianggap feasible.

Decoder juga memisahkan delay menjadi:

- `congestion_wait`: delay karena kapasitas machine,
- `tidal_wait`: delay tambahan karena constraint pasang surut.

Kalau decoder gagal menemukan operasi eligible atau tidal window yang valid, schedule dianggap infeasible. Runner lalu berhenti lewat `ensure_feasible(...)`, supaya hasil yang tersimpan bukan schedule yang diam-diam invalid.

## Constraint Pasang Surut

Constraint tidal bersifat hard constraint.

Mode yang didukung:

- `alur`: `start_h` harus berada di arrival window dan `end_h` harus berada di departure window.
- `sandar`: interval proses `[start_h, end_h]` harus overlap dengan minimal satu raw feasible window.

Machine tanpa tidal constraint selalu feasible. Untuk machine tidal, constraint hanya diterapkan pada `ship_name` yang terdaftar pada constraint machine tersebut.

Perbedaan penting:

- Mode `alur` cocok untuk constraint kapal masuk/keluar alur: awal operasi dan akhir operasi harus jatuh pada window yang benar.
- Mode `sandar` cocok untuk constraint kapal saat berada di pelabuhan: interval operasi cukup overlap dengan window pasang surut yang feasible.

## Objective dan Metrics

Objective utama:

```text
total_tardiness = sum(max(0, C_j - d_j))
d_j = first_arrival_j + T_j
```

Artinya, due time absolut tiap voyage dihitung dari arrival pertama voyage ditambah target window `T_j`. Jika voyage selesai sebelum due time, tardiness-nya nol. Jika selesai setelah due time, selisihnya masuk ke total tardiness.

Metrics output:

- `total_tardiness`
- `max_tardiness`
- `late_voyage_count`
- `is_feasible`
- `infeasible_reason`
- `penalty_tardiness`

Schedule infeasible diberi penalti `1e12` melalui `build_infeasible_metrics(...)`.

`is_feasible = true` berarti schedule lolos semua constraint yang diketahui decoder dan baseline runtime. `infeasible_reason` akan terisi jika ada kegagalan, misalnya tidal window tidak ditemukan.

## Output

Setiap runner menyimpan hasil ke direktori masing-masing:

```text
data/results/caoa/
data/results/caoassr/
data/results/gwo/
```

File utama per algoritma:

- `<algorithm>_optimized_timetable.csv`
- `<algorithm>_optimized_metrics.json`
- `<algorithm>_best_position.npy`
- `<algorithm>_convergence_curve.npy`
- `<algorithm>_convergence_curve.json`
- `<algorithm>_voyage_debug_report.csv`
- `<algorithm>_voyage_debug_summary.json`
- `fcfs_baseline_timetable.csv`
- `fcfs_baseline_metrics.json`
- `fcfs_vs_<algorithm>_voyage_comparison.csv`
- `fcfs_vs_<algorithm>_operation_comparison.csv`

Khusus CAOASSR:

- `caoassr_rdk_logs.json`
- `caoassr_rdk_summary.json`

Cara membaca file output:

- Timetable menjawab “operasi dijadwalkan kapan dan di machine mana?”
- Metrics menjawab “seberapa baik hasilnya secara objective?”
- Best position menyimpan vektor prioritas terbaik yang ditemukan optimizer.
- Convergence curve menunjukkan perkembangan objective selama iterasi.
- Voyage debug report menjelaskan keterlambatan per voyage.
- FCFS comparison menunjukkan perubahan schedule dari baseline ke optimizer.
- RDK logs khusus CAOASSR menjelaskan aktivitas guidance SSR/RDK selama run.

## CAOA, CAOASSR, dan GWO

Semua optimizer bekerja pada representasi yang sama: vektor prioritas kontinu sepanjang jumlah operasi.

`CAOA` adalah optimizer utama. Ia mencari prioritas operasi yang meminimasi tardiness dengan update populasi berbasis parameter `alpha`, `beta`, `gamma`, `delta`, dan `initial_energy`.

`CAOASSR` adalah varian CAOA yang menambahkan guidance. Tujuannya membantu pencarian ketika populasi mulai stagnan atau ketika elite solution memberi sinyal urutan operasi yang berguna. Diagnostic-nya disimpan sebagai RDK logs dan summary.

`GWO` dipakai sebagai pembanding. Karena memakai decoder dan objective yang sama, hasilnya bisa dibandingkan lebih adil dengan CAOA dan CAOASSR.

## Debugging Cepat

Jika hasil terasa aneh, cek berurutan:

1. `data/results/<algorithm>/<algorithm>_optimized_metrics.json`
2. `data/results/<algorithm>/<algorithm>_voyage_debug_report.csv`
3. `data/results/<algorithm>/fcfs_vs_<algorithm>_voyage_comparison.csv`
4. `data/results/<algorithm>/fcfs_vs_<algorithm>_operation_comparison.csv`
5. `data/results/<algorithm>/<algorithm>_convergence_curve.json`

Pertanyaan yang biasanya perlu dijawab:

- Apakah `is_feasible` bernilai `true`?
- Voyage mana yang paling terlambat?
- Apakah optimizer benar-benar memperbaiki FCFS?
- Apakah delay datang dari `congestion_wait` atau `tidal_wait`?
- Apakah convergence curve membaik atau stagnan sejak awal?

## Hasil Saat Ini

Ringkasan artefak metrics yang tersedia di branch ini:

| Method | Total Tardiness | Max Tardiness | Late Voyages | Feasible |
| --- | ---: | ---: | ---: | --- |
| FCFS baseline | 7761.0 | 1732.0 | 40 | true |
| CAOA | 7598.0 | 1732.0 | 40 | true |
| CAOASSR | 7585.0 | 1732.0 | 39 | true |
| GWO | 7593.0 | 1732.0 | 41 | true |

Pada artefak saat ini, CAOASSR menghasilkan `total_tardiness` terendah.
