import pandas as pd
import numpy as np

df = pd.read_csv('cumulative.csv')

# keep only confirmed planets and false positives
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# target column
df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)

# drop leakage cols (NASA's own output labels / FP flags)
leakage_cols = [
    'koi_disposition',
    'koi_pdisposition',
    'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# drop metadata / identifier cols
meta_cols = ['kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname', 'rowid']
df = df.drop(columns=[c for c in meta_cols if c in df.columns])

# drop 100% missing cols
df = df.drop(columns=['koi_teq_err1', 'koi_teq_err2'], errors='ignore')

# keep only numeric columns
df = df.select_dtypes(include='number')

# NO imputation — NaN values preserved intentionally
df.to_csv('kepler_clean_v2.csv', index=False)

print(f'Shape: {df.shape[0]} rows x {df.shape[1]} cols')
print(f'\nClass distribution:')
print(df['target'].value_counts().to_string())
print(f'\nNaN counts per column (only showing cols with NaN):')
nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
if len(nan_counts) == 0:
    print('  (none)')
else:
    for col, n in nan_counts.items():
        print(f'  {col}: {n} ({100*n/len(df):.1f}%)')
print(f'\nTotal NaN cells: {df.isna().sum().sum()}')
print(f'\nColumns ({len(df.columns)}):')
for col in df.columns:
    print(f'  {col}')
