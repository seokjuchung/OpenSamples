import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from microboone_utils import *
from file import File
import glob
import os

# Directory containing the files
data_dir = "/nevis/westside/data/sc5303/Data/ubOpen/h5/"
file_pattern = os.path.join(data_dir, "*.h5")
file_list = glob.glob(file_pattern)

all_primaries = []

tables = ['particle_table']

for file_path in file_list:
    f = File(file_path)
    for t in tables:
        f.add_group(t)
    f.read_data(0, len(f))
    parts = f.get_dataframe_evt('particle_table')
    # primaries = parts.loc[parts['start_process'] == b'primary'].copy()
    primaries = parts
    all_primaries.append(primaries)

# Remove empty DataFrames
all_primaries = [df for df in all_primaries if df is not None and not df.empty]
if len(all_primaries) == 0:
    print("No primary particles found in any files.")
    exit(0)

# Normalize columns: decode bytes -> str and convert categorical dtypes to string/object
for i, df in enumerate(all_primaries):
    for col in df.columns:
        try:
            # If column has numpy byte-string dtype (e.g. |S64), coerce elements to Python str
            kind = getattr(df[col].dtype, 'kind', None)
            if kind == 'S':
                df.loc[:, col] = df[col].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)

            # decode bytes in object columns
            if df[col].dtype == object:
                if df[col].apply(lambda x: isinstance(x, (bytes, bytearray))).any():
                    df.loc[:, col] = df[col].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)

            # convert categorical to string to avoid mixed-category concat issues
            if pd.api.types.is_categorical_dtype(df[col]):
                df.loc[:, col] = df[col].astype(str)
        except Exception:
            # if any column coercion fails, continue — we'll surface types below if concat fails
            pass
    all_primaries[i] = df

# Concatenate all primaries DataFrames with a safety wrapper
try:
    all_primaries_df = pd.concat(all_primaries, ignore_index=True, sort=False)
except Exception as e:
    print("Failed to concat DataFrames. Printing dtypes for diagnostics:")
    for idx, df in enumerate(all_primaries):
        print(f"--- DataFrame {idx} (length={len(df)}) dtypes ---")
        print(df.dtypes)
        print(df.head(3))
    raise

# Print all unique g4_pdg values in the primaries DataFrame
unique_pdg, counts = np.unique(all_primaries_df['g4_pdg'], return_counts=True)
print("Unique g4_pdg values and their frequencies:")
for pdg, count in zip(unique_pdg, counts):
    print(f"  {pdg}: {count}")

# Prepare data for stacked histogram: group by g4_pdg and category
pdg_values = all_primaries_df['g4_pdg'].unique()
cat_values = all_primaries_df['category'].unique()

# Create a 2D array: rows = categories, columns = pdg values
hist_data = []
for cat in cat_values:
    hist_data.append(all_primaries_df[all_primaries_df['category'] == cat]['g4_pdg'].values)

# Remove entries where abs(g4_pdg) > 1e4
hist_data = [data[(np.abs(data) <= 1e4)] for data in hist_data]

plt.figure()
plt.hist(hist_data, bins=50, stacked=True, label=[f'category {cat}' for cat in cat_values])
plt.yscale('log')
plt.xlabel('g4_pdg')
plt.ylabel('Count (log scale)')
# plt.xlim([-5e3, 5e3])
plt.title('Histogram of g4_pdg for Primary Particles (stacked by category, all files)')
plt.legend()
plt.savefig('plots/g4_pdg_histogram_all_files.png')
plt.close()
