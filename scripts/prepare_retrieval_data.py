"""
One-time script: downloads the raw dataset, builds the resolved_pairs corpus,
and saves it to disk — so the deployed app never needs Kaggle access itself,
just this pre-built file.
"""

import sys
sys.path.insert(0, 'src')

import os
import kagglehub
import pandas as pd

from data_prep import load_raw_dataset, build_spotify_subsets

kagglehub.login()

path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")

print(f"Dataset at: {path}")

df = load_raw_dataset(f"{path}/twcs/twcs.csv")

print(f"Loaded {len(df)} total rows")

subsets = build_spotify_subsets(df)

resolved_pairs = subsets['resolved_pairs']

print(f"Built resolved_pairs: {len(resolved_pairs)} rows")

os.makedirs('data/processed', exist_ok=True)

resolved_pairs.to_csv(
    'data/processed/resolved_pairs.csv',
    index=False
)

print("Saved to data/processed/resolved_pairs.csv")
