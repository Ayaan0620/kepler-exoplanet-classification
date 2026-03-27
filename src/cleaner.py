import pandas as pd
import numpy as np

df = pd.read_csv('../data/cumulative.csv')

# only keep confirmed and false positives, drop candidates
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# target: 1 = confirmed planet, 0 = false positive
df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)

# --- drop leaky columns ---
# these are basically NASA's own classification outputs
leakage_cols = [
    'koi_disposition',
    'koi_pdisposition',
    'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# drop id/metadata stuff
meta_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname', 'rowid']
df = df.drop(columns=[c for c in meta_cols if c in df.columns])

# these two columns are completely empty
df = df.drop(columns=['koi_teq_err1', 'koi_teq_err2'], errors='ignore')

# only numeric
df = df.select_dtypes(include='number')

# NOTE: we keep NaN values here on purpose
# imputation happens inside the CV loop to avoid data leakage
df.to_csv('../data/kepler_clean_v2.csv', index=False)

print(f'Shape: {df.shape[0]} rows x {df.shape[1]} cols')
print(f'\nClass distribution:')
print(df['target'].value_counts().to_string())
nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
print(f'\nColumns with NaN:')
if len(nan_counts) == 0:
    print('  none')
else:
    for col, n in nan_counts.items():
        print(f'  {col}: {n} ({100*n/len(df):.1f}%)')
print(f'\nTotal NaN cells: {df.isna().sum().sum()}')
