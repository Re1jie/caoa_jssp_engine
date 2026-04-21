import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/voyage_data.csv")

T_ZERO = pd.to_datetime('2025-01-01 00:00:00')

month_map = {
    'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr', 'Mei': 'May', 'Jun': 'Jun',
    'Jul': 'Jul', 'Agu': 'Aug', 'Sep': 'Sep', 'Okt': 'Oct', 'Nov': 'Nov', 'Des': 'Dec'
}
df['ETA_TANGGAL'] = df['ETA_TANGGAL'].replace(month_map, regex=True)
df['ETD_TANGGAL'] = df['ETD_TANGGAL'].replace(month_map, regex=True)

df['ETA'] = pd.to_datetime(df['ETA_TANGGAL'] + ' ' + df['ETA_JAM'], format='%d-%b-%y %H:%M')
df['ETD'] = pd.to_datetime(df['ETD_TANGGAL'] + ' ' + df['ETD_JAM'], format='%d-%b-%y %H:%M')

df['A_lj'] = (df['ETA'] - T_ZERO).dt.total_seconds() / 3600
df['T_lj'] = (df['ETD'] - T_ZERO).dt.total_seconds() / 3600

df = df.dropna(subset=['A_lj', 'T_lj'])

df['p_lj'] = (df['T_lj'] - df['A_lj']).fillna(0)
df = df.sort_values(by=['NAMA_KAPAL', 'A_lj']).reset_index(drop=True)
df['TSail_lj'] = df.groupby('NAMA_KAPAL')['A_lj'].shift(-1) - df['T_lj']

df['PELABUHAN_LOGIS'] = np.where(df['p_lj'] > 90, 'D_' + df['PELABUHAN'], df['PELABUHAN'])

unique_ships = df['NAMA_KAPAL'].unique()
job_map = {ship: idx for idx, ship in enumerate(unique_ships)}

unique_ports = df['PELABUHAN_LOGIS'].unique()
machine_map = {port: idx for idx, port in enumerate(unique_ports)}

df['job_id'] = df['NAMA_KAPAL'].map(job_map)
df['machine_id'] = df['PELABUHAN_LOGIS'].map(machine_map)

df = df.sort_values(by=['job_id', 'A_lj']).reset_index(drop=True)
df['op_seq'] = df.groupby('job_id').cumcount()

# Sentralisasi Aturan Rute Kapal
route_rules = {
    'KM.DOBONSOLO': (['BIAK', 'JAYAPURA'], 'A', 'B'),
    'KM.KELIMUTU': (['BATULICIN', 'SURABAYA'], 'B', 'A'),
    'KM.DOROLONDA': (['WAREN', 'NAMLEA'], 'B', 'A'),
    'KM.LAWIT': (['BENOA', 'BIMA', 'WAINGAPU', 'ENDE', 'KALABAHI', 'KUPANG', 'ROTE'], 'B', 'A'),
    'KM.BINAIYA': (['WAINGAPU', 'ENDE', 'KUPANG'], 'B', 'A'),
    'KM.TATAMAILAU': (['TERNATE', 'AMBON'], 'B', 'A')
}

df['rute'] = 'default'
for kapal, (ports, true_route, false_route) in route_rules.items():
    mask = df['NAMA_KAPAL'] == kapal
    if mask.any():
        has_special_port = df[mask].groupby('job_id')['PELABUHAN_LOGIS'].transform(lambda x: x.isin(ports).any())
        df.loc[mask, 'rute'] = has_special_port.map({True: true_route, False: false_route})

def apply_LDR(group):
    layer_id, visited = 0, set()
    layers = []
    for machine in group['machine_id']:
        if machine in visited:
            layer_id += 1
            visited = {machine}
        else:
            visited.add(machine)
        layers.append(layer_id)
    group = group.copy()
    group['layer_id'] = layers
    return group

df = pd.concat([apply_LDR(g) for _, g in df.groupby('job_id')], ignore_index=True)

port_region_map = {
    "SURABAYA": "Barat", "D_SURABAYA": "Barat", "SEMARANG": "Barat", "D_SEMARANG": "Barat",
    "TANJUNG PRIOK": "Barat", "D_TANJUNG PRIOK": "Barat", "D_CIREBON": "Barat", "D_CILEGON": "Barat",
    "KUMAI": "Barat", "SAMPIT": "Barat", "PONTIANAK": "Barat", "BATULICIN": "Barat",
    "BATAM": "Barat", "TANJUNG BALAI KARIMUN": "Barat", "BELAWAN": "Barat",
    "KIJANG": "Barat", "LETUNG": "Barat", "TAREMPA": "Barat", "NATUNA": "Barat", 
    "MIDAI": "Barat", "SERASAN": "Barat", "BLINYU": "Barat", "TANJUNG PANDAN": "Barat", 
    "KARIMUN JAWA": "Barat",
    "MAKASSAR": "Tengah", "D_MAKASSAR": "Tengah", "KENDARI": "Tengah", "D_KENDARI": "Tengah",
    "BAUBAU": "Tengah", "D_BAUBAU": "Tengah", "WANCI": "Tengah", "RAHA": "Tengah", 
    "PAREPARE": "Tengah", "AWERANGE": "Tengah", "PANTOLOAN": "Tengah", "BITUNG": "Tengah", 
    "BANGGAI": "Tengah", "LUWUK": "Tengah", "GORONTALO": "Tengah", "BALIKPAPAN": "Tengah", 
    "TARAKAN": "Tengah", "NUNUKAN": "Tengah", "BONTANG": "Tengah", "BENOA": "Tengah", 
    "BIMA": "Tengah", "WAINGAPU": "Tengah", "ENDE": "Tengah", "KUPANG": "Tengah", 
    "KALABAHI": "Tengah", "LABUAN BAJO": "Tengah", "MAUMERE": "Tengah", "LEWOLEBA": "Tengah", 
    "LEMBAR": "Tengah", "LARANTUKA": "Tengah", "ROTE": "Tengah", "WAIKELO": "Tengah",
    "AMBON": "Timur", "SORONG": "Timur", "SERUI": "Timur", "JAYAPURA": "Timur", 
    "MANOKWARI": "Timur", "BIAK": "Timur", "NAMLEA": "Timur", "NABIRE": "Timur", 
    "KAIMANA": "Timur", "FAKFAK": "Timur", "DOBO": "Timur", "TUAL": "Timur", 
    "BANDA": "Timur", "TERNATE": "Timur", "WAREN": "Timur", "WASIOR": "Timur", 
    "NAMROLE": "Timur", "SAUMLAKI": "Timur", "TIMIKA": "Timur", "AGATS": "Timur", 
    "MERAUKE": "Timur", "JAILOLO": "Timur", "BACAN": "Timur", "SANANA": "Timur", 
    "GESER": "Timur", "TIDORE": "Timur"
}

df['wilayah_pelabuhan'] = df['PELABUHAN_LOGIS'].map(port_region_map)

dominant_region = df.groupby('job_id')['wilayah_pelabuhan'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
df['wilayah_kapal'] = df['job_id'].map(dominant_region)

komputasi_df = df[['job_id', 'machine_id', 'op_seq', 'rute', 'layer_id', 'A_lj', 'p_lj', 'TSail_lj', 'wilayah_kapal']]
df['voyage'] = df['VOYAGE'].astype(str).str.split('.').str[0].astype(int)
komputasi_df = df[['job_id', 'voyage', 'machine_id', 'op_seq', 'rute', 'layer_id', 'A_lj', 'p_lj', 'TSail_lj', 'wilayah_kapal']]
komputasi_df.to_csv("data/processed/jssp_data.csv", index=False)

dual_berth = {"TANJUNG PRIOK": 1, "MAKASSAR": 1, "BAUBAU": 1, "AMBON": 1, "SORONG": 1, "KUPANG": 1, "SURABAYA": 1}

master_df = df[['PELABUHAN_LOGIS', 'machine_id']].drop_duplicates().sort_values('machine_id').reset_index(drop=True)
master_df['num_berth'] = master_df['PELABUHAN_LOGIS'].str.replace(r'^D_', '', regex=True).map(dual_berth).fillna(1).astype(int)
master_df.to_csv("data/processed/machine_master.csv", index=False)

SK_Trayek = {
    "KM.KELUD": {"default": 168.0}, "KM.BUKITRAYA": {"default": 336.0}, "KM.LABOBAR": {"default": 336.0},
    "KM.GUNUNGDEMPO": {"default": 336.0}, "KM.TIDAR": {"default": 336.0}, "KM.NGGAPULU": {"default": 336.0},
    "KM.CIREMAI": {"default": 336.0}, "KM.SINABUNG": {"default": 336.0}, "KM.AWU": {"default": 336.0},
    "KM.LEUSER": {"default": 672.0}, "KM.EGON": {"default": 336.0}, "KM.TILONGKABILA": {"default": 336.0},
    "KM.SIRIMAU": {"default": 672.0}, "KM.WILIS": {"default": 336.0}, "KM.LAMBELU": {"default": 336.0},
    "KM.BUKITSIGUNTANG": {"default": 336.0}, "KFC.JETLINER": {"default": 168.0},
    "KM.SANGIANG": {"default": 336.0}, "KM.PANGRANGO": {"default": 336.0},
    "KM.DOBONSOLO": {"A": 384.0, "B": 288.0},
    "KM.KELIMUTU": {"A": 288.0, "B": 384.0},
    "KM.DOROLONDA": {"A": 336.0, "B": 336.0},
    "KM.LAWIT": {"A": 336.0, "B": 336.0},
    "KM.BINAIYA": {"A": 336.0, "B": 336.0},
    "KM.TATAMAILAU": {"A": 336.0, "B": 336.0},
}

job_route_df = df[['job_id', 'voyage', 'NAMA_KAPAL', 'rute']].drop_duplicates().reset_index(drop=True)
job_route_df['T_j'] = job_route_df.apply(lambda row: SK_Trayek.get(row['NAMA_KAPAL'], {}).get(row['rute'], np.nan), axis=1)

final_job_df = job_route_df[['job_id', 'voyage', 'T_j']]
final_job_df.to_csv("data/processed/job_target_time.csv", index=False)