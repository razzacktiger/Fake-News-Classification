"""
Preprocessing script for ISOT dataset.

This script:
1. Loads ISOT dataset (True.csv, Fake.csv)
2. Combines title and text
3. Splits into train/valid/test (70/15/15)
4. Saves preprocessed datasets to data/processed/
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split


def preprocess_isot():
    """Main preprocessing function for ISOT dataset."""
    print("=" * 60)
    print("ISOT Dataset Preprocessing")
    print("=" * 60)
    
    # Load ISOT datasets
    print("\nLoading ISOT datasets...")
    isot_true = pd.read_csv('data/raw/True.csv')
    isot_fake = pd.read_csv('data/raw/Fake.csv')
    
    print(f"Original shapes - True: {isot_true.shape}, Fake: {isot_fake.shape}")
    
    # Add labels (1 = true, 0 = fake)
    isot_true['label'] = 1
    isot_fake['label'] = 0
    
    # Combine true and fake
    isot_df = pd.concat([isot_true, isot_fake], ignore_index=True)
    
    # Combine title and text for better features
    isot_df['text'] = isot_df['title'].fillna('') + ' ' + isot_df['text'].fillna('')
    isot_df['text'] = isot_df['text'].str.strip()
    isot_df['text'].fillna('', inplace=True)
    
    # Select only relevant columns
    isot_df = isot_df[['text', 'label']]
    
    print(f"\nCombined ISOT dataset shape: {isot_df.shape}")
    print(f"Label distribution:\n{isot_df['label'].value_counts()}")
    
    # Split ISOT into train/valid/test (70/15/15)
    print("\nSplitting into train/valid/test (70/15/15)...")
    isot_train, isot_temp = train_test_split(
        isot_df, test_size=0.3, stratify=isot_df['label'], random_state=42
    )
    isot_valid, isot_test = train_test_split(
        isot_temp, test_size=0.5, stratify=isot_temp['label'], random_state=42
    )
    
    print(f"\nAfter splitting:")
    print(f"  Train: {isot_train.shape}")
    print(f"  Valid: {isot_valid.shape}")
    print(f"  Test:  {isot_test.shape}")
    print(f"\nTrain label distribution:\n{isot_train['label'].value_counts()}")
    
    # Save preprocessed datasets
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    isot_train.to_csv(f'{output_dir}/isot_train.csv', index=False)
    isot_valid.to_csv(f'{output_dir}/isot_valid.csv', index=False)
    isot_test.to_csv(f'{output_dir}/isot_test.csv', index=False)
    
    print(f"\nSaved preprocessed ISOT datasets to {output_dir}/")
    print(f"  - isot_train.csv")
    print(f"  - isot_valid.csv")
    print(f"  - isot_test.csv")
    print("\n" + "=" * 60)
    print("ISOT preprocessing completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    preprocess_isot()

