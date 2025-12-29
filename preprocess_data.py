import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from microboone_utils import *
from file import File
import os
import glob
import matplotlib.pyplot as plt

data_dir = "/nevis/westside/data/sc5303/Data/ubOpen/h5/"
files = glob.glob(os.path.join(data_dir, "*.h5"))

if len(files) == 0:
    print("No .h5 files found in", data_dir)
    exit(0)

# ensure output dir exists
os.makedirs("plots", exist_ok=True)

all_min_wire = []
all_max_wire = []
all_nu_wire = []
all_min_time = []
all_max_time = []
all_nu_time = []

tables = ['hit_table', 'edep_table', 'particle_table', 'event_table']

for file_path in files:
    try:
        f = File(file_path)
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
        continue
    nevts = len(f)
    for t in tables:
        f.add_group(t)
    f.read_data(0, nevts)
    evts = f.build_evt()

    for evt in evts:
        evtevts = evt['event_table']
        evthits = evt['hit_table']
        evtdeps = evt['edep_table']
        evtparts = evt['particle_table']

        evtdeps = evtdeps.sort_values('energy_fraction', ascending=False, kind='mergesort').drop_duplicates("hit_id")
        evthits = evthits.merge(evtdeps, on="hit_id", how="left")
        evthits['g4_id'] = evthits['g4_id'].fillna(-1)
        evthits = evthits.fillna(0)
        evthits["cosmic_label"] = np.where(evthits["g4_id"] < 0, 'cosmic', 'neutrino')
        evthits = evthits.merge(evtparts[['g4_id', 'category', 'instance']], on="g4_id", how="left")
        evthits["category_name"] = 'cosmic'
        for l in category:
            evthits.loc[evthits["category"] == l.value, "category_name"] = l.name

        filtered = evthits[(evthits['local_plane'] == 2) & (~evthits['category_name'].isin(['cosmic', 'other']))]

        if not filtered.empty:
            all_min_wire.append(filtered['local_wire'].min())
            all_max_wire.append(filtered['local_wire'].max())
            all_min_time.append(filtered['local_time'].min())
            all_max_time.append(filtered['local_time'].max())
            all_nu_wire.append(evtevts['nu_vtx_wire_pos_2'].iloc[0])
            all_nu_time.append(evtevts['nu_vtx_wire_time'].iloc[0])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(all_min_wire, bins=50, alpha=0.5, label='Min Wire')
plt.hist(all_max_wire, bins=50, alpha=0.5, label='Max Wire')
plt.hist(all_nu_wire, bins=50, alpha=0.5, label='Neutrino Wire')
plt.xlabel('Wire Value')
plt.ylabel('Count')
plt.title('Wire Value Distributions')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(all_min_time, bins=50, alpha=0.5, label='Min Time')
plt.hist(all_max_time, bins=50, alpha=0.5, label='Max Time')
plt.hist(all_nu_time, bins=50, alpha=0.5, label='Neutrino Time')
plt.xlabel('Time Value')
plt.ylabel('Count')
plt.title('Time Value Distributions')
plt.legend()

plt.tight_layout()
plt.savefig("plots/wire_time_distributions.png")
