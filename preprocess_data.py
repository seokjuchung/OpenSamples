import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from microboone_utils import *
from file import File
import os
import glob
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

data_dir = "/nevis/westside/data/sc5303/Data/ubOpen/h5/"
files = glob.glob(os.path.join(data_dir, "*.h5"))
files = files[:2]  # limit to first 100 files for testing

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

def process_file(file_path, tables):
    """Process a single file and return per-file aggregated lists."""
    min_wire = []
    max_wire = []
    nu_wire = []
    min_time = []
    max_time = []
    nu_time = []
    try:
        f = File(file_path)
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")
        return min_wire, max_wire, nu_wire, min_time, max_time, nu_time

    nevts = len(f)
    for t in tables:
        f.add_group(t)
    try:
        f.read_data(0, nevts)
        evts = f.build_evt()
    except Exception as e:
        print(f"Error reading events from {file_path}: {e}")
        return min_wire, max_wire, nu_wire, min_time, max_time, nu_time

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
            min_wire.append(filtered['local_wire'].min())
            max_wire.append(filtered['local_wire'].max())
            min_time.append(filtered['local_time'].min())
            max_time.append(filtered['local_time'].max())
            # guard access in case event table missing fields
            try:
                nu_wire.append(evtevts['nu_vtx_wire_pos_2'].iloc[0])
            except Exception:
                nu_wire.append(np.nan)
            try:
                nu_time.append(evtevts['nu_vtx_wire_time'].iloc[0])
            except Exception:
                nu_time.append(np.nan)

    return min_wire, max_wire, nu_wire, min_time, max_time, nu_time


# Determine worker count (threads) — threads are chosen because HDF5+MPI can misbehave with forked processes.
max_workers = min(32, max(1, multiprocessing.cpu_count(), len(files)))

futures = []
if tqdm is not None:
    iterator = files
else:
    iterator = files

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for file_path in iterator:
        futures.append(ex.submit(process_file, file_path, tables))

    if tqdm is not None:
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                a, b, c, d, e, f_ = fut.result()
            except Exception as exc:
                print(f"File worker raised: {exc}")
                continue
            all_min_wire.extend(a)
            all_max_wire.extend(b)
            all_nu_wire.extend(c)
            all_min_time.extend(d)
            all_max_time.extend(e)
            all_nu_time.extend(f_)
    else:
        for fut in as_completed(futures):
            try:
                a, b, c, d, e, f_ = fut.result()
            except Exception as exc:
                print(f"File worker raised: {exc}")
                continue
            all_min_wire.extend(a)
            all_max_wire.extend(b)
            all_nu_wire.extend(c)
            all_min_time.extend(d)
            all_max_time.extend(e)
            all_nu_time.extend(f_)

plt.figure(figsize=(12, 5))

# Prepare difference arrays for plotting
min_wire_arr = np.array(all_min_wire)
max_wire_arr = np.array(all_max_wire)
nu_wire_arr = np.array(all_nu_wire)

min_time_arr = np.array(all_min_time)
max_time_arr = np.array(all_max_time)
nu_time_arr = np.array(all_nu_time)

# Wire: nu - min, and max - nu
mask_w1 = np.isfinite(nu_wire_arr) & np.isfinite(min_wire_arr)
mask_w2 = np.isfinite(max_wire_arr) & np.isfinite(nu_wire_arr)
wire_nu_minus_min = nu_wire_arr[mask_w1] - min_wire_arr[mask_w1]
wire_max_minus_nu = max_wire_arr[mask_w2] - nu_wire_arr[mask_w2]

plt.subplot(1, 2, 1)
plt.hist(wire_nu_minus_min, bins=50, alpha=0.6, label='nu - min (wire)')
plt.hist(wire_max_minus_nu, bins=50, alpha=0.6, label='max - nu (wire)')
plt.xlabel('Wire difference')
plt.ylabel('Count')
plt.title('Wire: nu-min and max-nu')
plt.legend()

# Time: nu - min_time, and nu - max_time
mask_t1 = np.isfinite(nu_time_arr) & np.isfinite(min_time_arr)
mask_t2 = np.isfinite(nu_time_arr) & np.isfinite(max_time_arr)
time_nu_minus_min = nu_time_arr[mask_t1] - min_time_arr[mask_t1]
time_nu_minus_max = nu_time_arr[mask_t2] - max_time_arr[mask_t2]

plt.subplot(1, 2, 2)
plt.hist(time_nu_minus_min, bins=50, alpha=0.6, label='nu - min (time)')
plt.hist(time_nu_minus_max, bins=50, alpha=0.6, label='nu - max (time)')
plt.xlabel('Time difference')
plt.ylabel('Count')
plt.title('Time: nu-min and nu-max')
plt.legend()

plt.tight_layout()
plt.savefig("plots/wire_time_distributions.png")
