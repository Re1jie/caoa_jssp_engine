# Pipeline Review

Dokumen ini merangkum pipeline runtime yang aktif saat ini, dari data hasil preprocessing sampai output optimisasi.

## Ringkasannya

Pipeline berjalan seperti ini:

1. Runner memuat data operasi slice, target due, dan kapasitas machine dari `data/processed/`.
2. `TidalChecker` memuat constraint pasang surut per `machine_id`, termasuk mode `alur` dan `sandar`.
3. Baseline FCFS dibangun sebagai pembanding awal.
4. CAOA mengoptimasi vektor prioritas kontinu berdimensi `jumlah_operasi`.
5. Decoder aktif mengubah vektor itu menjadi schedule aktual yang patuh precedence, kapasitas berth, sailing time, dan constraint tidal aktif.
6. Schedule dihitung metriknya dengan objective aktif `total_tardiness`.
7. Jika hasil final infeasible, runner menghentikan proses dengan hard failure.
8. Jika feasible, timetable, metrics, best position, dan debug report voyage disimpan ke `data/result/`.

## Objective dan metrik aktif

Objective runtime saat ini adalah `standard total tardiness`:

`total_tardiness = sum(max(0, C_j - d_j))`

dengan:

- `C_j`: completion time operasi terakhir per voyage
- `d_j`: due absolut voyage = `first_arrival + T_j`

Metrik utama yang dipakai di runtime:

- `total_tardiness`
- `max_tardiness`
- `late_voyage_count`
- `is_feasible`
- `infeasible_reason`
- `penalty_tardiness`

Jika schedule infeasible, sistem memakai penalti `1e12` melalui `build_infeasible_metrics(...)`.

## Input runtime

Loader utama: [utils/data_loader.py](/home/re1jie/caoa_jssp_engine/utils/data_loader.py:1)

File yang dibaca saat run:

- `data/processed/jssp_data_sliced.csv`
- `data/processed/job_target_time_sliced.csv`
- `data/processed/machine_master.csv`
- `data/processed/tidal_constraints.csv`
- `data/processed/tidal_feasible_windows.csv`
- `data/processed/tidal_hourly_lookup.csv`

Struktur penting:

- `df_ops`: `job_id`, `voyage`, `op_seq`, `machine_id`, `A_lj`, `p_lj`, `TSail_lj`, opsional `ship_name`
- `df_machine_master`: `machine_id`, `num_berth`
- `df_job_target`: `job_id`, `voyage`, `T_j`, opsional `ship_name`

Loader juga memvalidasi bahwa pasangan `(job_id, voyage)` di operasi dan target harus identik.

## Runner utama

Entry point aktif: [run_insertion.py](/home/re1jie/caoa_jssp_engine/run_insertion.py:1)

Urutan kerjanya:

1. Set seed `numpy`.
2. Load `df_ops`, `df_machine_master`, `df_job_target`.
3. Inisialisasi `TidalChecker()`.
4. Jalankan baseline FCFS via `run_fcfs_baseline(...)`.
5. Panggil `ensure_feasible(...)` untuk memaksa baseline feasible.
6. Bangun `ActiveScheduleDecoder` dari `engine/decoder_insertion.py`.
7. Jalankan `CAOA(...)` untuk mencari `best_position`.
8. Decode `best_position` menjadi schedule final.
9. Panggil `ensure_feasible(...)` lagi untuk hasil CAOA.
10. Simpan hasil optimisasi dan debug report voyage.

Catatan penting:

- Runner aktif memakai [engine/decoder_insertion.py](/home/re1jie/caoa_jssp_engine/engine/decoder_insertion.py:1), bukan `engine/decoder.py`.
- `engine/decoder.py` masih ada, tetapi bukan decoder yang dipakai entry point utama saat ini.

## Baseline FCFS

Implementasi: [engine/fcfs.py](/home/re1jie/caoa_jssp_engine/engine/fcfs.py:1)

Karakter baseline:

- event-driven
- urutan antrean per machine bersifat FCFS
- tetap menghormati precedence antar operasi voyage
- next operation dirilis pada `completion_prev + TSail_lj`
- kapasitas machine mengikuti `num_berth`
- tidal constraint dicek sebelum start operasi

Jika berth tersedia, operasi langsung dicoba dijadwalkan. Jika tidal aktif, start digeser ke feasible window berikutnya. Jika tidak ada window feasible, baseline langsung mengembalikan metrics infeasible.

Interpretasi tidal yang dipakai baseline:

- `mode='alur'`: start operasi harus memenuhi arrival window dan completion harus memenuhi departure window.
- `mode='sandar'`: interval proses `[S_lj, C_lj]` harus overlap dengan minimal satu raw feasible window, artinya cukup ada satu jam/titik tidal yang memenuhi `E_min` selama kapal sandar.

## Decoder aktif untuk CAOA

Implementasi aktif: [engine/decoder_insertion.py](/home/re1jie/caoa_jssp_engine/engine/decoder_insertion.py:1)

Peran decoder:

- menerima vektor kontinu `X`
- memetakan setiap elemen `X[i]` ke satu operasi pada `L_ref`
- memilih operasi eligible dengan prioritas terkecil
- mencari slot start paling awal yang feasible pada machine terkait

Berbeda dari decoder event-driven lama, decoder ini bekerja dengan pendekatan insertion pada timeline machine:

- hanya operasi berikutnya per voyage yang boleh eligible
- release operasi pertama = `A_lj`
- release operasi berikutnya = `C_lj(prev) + TSail_lj(prev)`
- kandidat dipilih berdasarkan `(priority, arrival_h, job_id, voyage, op_seq)`
- untuk machine tertentu, decoder mencari candidate start dari `release_h` dan setiap `end_h` interval yang sudah ada
- kapasitas dicek dengan overlap counting terhadap timeline machine
- tidal diterapkan saat mencari slot feasible akhir

Interpretasi tidal pada decoder sama persis dengan baseline:

- `mode='alur'` memakai arrival/departure windows
- `mode='sandar'` memakai overlap interval sandar terhadap raw feasible window

Waktu tunggu dipisah menjadi:

- `congestion_wait`: delay karena kapasitas machine
- `tidal_wait`: delay tambahan karena constraint pasang surut

Jika tidak ada operasi eligible atau tidal window tidak ditemukan, decoder mengembalikan schedule parsial beserta metrics infeasible.

## Tidal constraint

Implementasi: [engine/tidal_checker.py](/home/re1jie/caoa_jssp_engine/engine/tidal_checker.py:1)

`TidalChecker` memuat:

- daftar machine yang terkena constraint tidal
- ship list yang benar-benar terkena constraint per machine
- mode constraint per machine: `alur` atau `sandar`
- arrival/departure feasible windows untuk mode `alur`
- raw feasible windows untuk mode `sandar`
- hourly lookup untuk debugging

Aturan runtime:

- machine non-tidal selalu feasible
- machine tidal hanya membatasi kapal yang `ship_name`-nya terdaftar pada constraint machine tersebut
- untuk `mode='alur'`, sebuah operasi dianggap feasible bila:
  - `start_h` berada di arrival window
  - `end_h = start_h + duration` berada di departure window
- untuk `mode='sandar'`, sebuah operasi dianggap feasible bila:
  - interval `[start_h, end_h]` overlap dengan minimal satu raw feasible window
  - ekuivalen dengan adanya minimal satu nilai `is_feasible = true` selama kapal sandar
- `find_next_start(...)` mencari start feasible paling awal sesuai mode constraint machine tersebut

Jika tidak ada start feasible, sistem menganggap operasi gagal secara hard failure, bukan soft penalty tambahan selain penalti infeasible global.

## Optimizer CAOA

Implementasi: [engine/caoa.py](/home/re1jie/caoa_jssp_engine/engine/caoa.py:1)

CAOA mengoptimasi vektor prioritas kontinu dengan pola umum:

- inisialisasi populasi dalam rentang `[lb, ub]`
- evaluasi semua individu dengan objective decoder
- pilih leader secara probabilistik berdasar fitness
- update posisi kandidat
- re-randomize jika solusi memburuk tajam
- reset individu yang energinya habis
- simpan `gBest`

Pada runner aktif, parameter utamanya:

- `N = 20`
- `max_iter = 100`
- `lb = 0.0`
- `ub = 1.0`
- `alpha = 0.9`
- `beta = 0.1`
- `gamma = 0.07`
- `delta = 1.2`
- `initial_energy = 150`

Dimensi solusi selalu `len(df_ops)`.

## Perhitungan metrics

Implementasi: [engine/metrics.py](/home/re1jie/caoa_jssp_engine/engine/metrics.py:1)

Metrics dihitung per voyage:

1. group schedule berdasarkan `(job_id, voyage)`
2. ambil `first_arrival = min(A_lj)`
3. ambil `last_completion = max(C_lj)`
4. hitung `due = first_arrival + T_j`
5. hitung `tardiness = max(0, last_completion - due)`

Validasi tambahan:

- semua target harus muncul di schedule
- schedule tidak boleh memiliki voyage ekstra tanpa target

Pelanggaran terhadap dua kondisi itu melempar `ValueError`.

## Artefak output

Hasil akhir disimpan oleh `run_insertion.py` ke `data/result/`:

- `caoa_optimized_timetable.csv`
- `caoa_optimized_metrics.json`
- `caoa_best_position.npy`
- `caoa_voyage_debug_report.csv`
- `caoa_voyage_debug_summary.json`

Isi debug report voyage:

- `first_arrival_hour`
- `last_completion_hour`
- `due_window_hours`
- `due_hour_absolute`
- `tardiness_hours`
- `lateness_hours`
- `earliness_hours`
- `debug_status`

## Catatan perubahan terhadap versi lama

Perubahan penting yang sudah tercermin di codebase saat ini:

- decoder aktif sudah pindah ke model insertion/capacity-aware slot search
- objective runtime berfokus pada tardiness standar
- infeasibility tidal ditangani sebagai hard failure
- tidal sekarang mendukung dua mode hard constraint:
  - `alur` untuk aturan arrival/departure window
  - `sandar` untuk aturan overlap selama kapal diproses di pelabuhan
- runner mewajibkan FCFS dan hasil CAOA sama-sama feasible sebelum output dianggap valid
- debug report voyage dibangun langsung dari schedule final dan `T_j`

Secara singkat, pipeline terbaru sekarang lebih ketat pada feasibility dan lebih eksplisit memisahkan delay akibat antrean machine vs delay akibat tidal.
