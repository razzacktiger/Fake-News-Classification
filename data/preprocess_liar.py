"""
Preprocessing script for LIAR dataset.

This script:
1. Loads LIAR dataset (train.tsv, valid.tsv, test.tsv)
2. Converts labels to binary (0 = false, 1 = true)
3. Standardizes column names
4. Saves preprocessed datasets to data/processed/
"""

import pandas as pd
import os

# Column names for LIAR dataset
LIAR_COLUMN_NAMES = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
]


def create_binary_labels(df):
    """Convert LIAR multi-class labels to binary labels."""
    label_map = {
        'true': 1,
        'mostly-true': 1,
        'half-true': 0,  # Including half-true as false
        'false': 0,
        'pants-fire': 0,
        'barely-true': 0,
    }
    df_copy = df.copy()
    df_copy['label_binary'] = df_copy['label'].map(label_map)
    df_copy.dropna(subset=['label_binary'], inplace=True)
    df_copy['label_binary'] = df_copy['label_binary'].astype(int)
    df_copy['statement'].fillna('', inplace=True)
    return df_copy


def preprocess_liar():
    """Main preprocessing function for LIAR dataset."""
    print("=" * 60)
    print("LIAR Dataset Preprocessing")
    print("=" * 60)
    
    # Load LIAR datasets
    print("\nLoading LIAR datasets...")
    train_df = pd.read_csv('data/raw/train.tsv', sep='\t', names=LIAR_COLUMN_NAMES)
    valid_df = pd.read_csv('data/raw/valid.tsv', sep='\t', names=LIAR_COLUMN_NAMES)
    test_df = pd.read_csv('data/raw/test.tsv', sep='\t', names=LIAR_COLUMN_NAMES)
    
    print(f"Original shapes - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    
    # Convert labels to binary
    print("\nConverting labels to binary...")
    train_df = create_binary_labels(train_df)
    valid_df = create_binary_labels(valid_df)
    test_df = create_binary_labels(test_df)
    
    # Drop the original 'label' column (multi-class) before renaming
    train_df = train_df.drop(columns=['label'], errors='ignore')
    valid_df = valid_df.drop(columns=['label'], errors='ignore')
    test_df = test_df.drop(columns=['label'], errors='ignore')
    
    # Standardize column names
    train_df = train_df.rename(columns={'statement': 'text', 'label_binary': 'label'})
    valid_df = valid_df.rename(columns={'statement': 'text', 'label_binary': 'label'})
    test_df = test_df.rename(columns={'statement': 'text', 'label_binary': 'label'})
    
    # Select only relevant columns
    train_df = train_df[['text', 'label']].copy()
    valid_df = valid_df[['text', 'label']].copy()
    test_df = test_df[['text', 'label']].copy()
    
    # Ensure label column is a simple integer Series
    train_df['label'] = train_df['label'].astype(int)
    valid_df['label'] = valid_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    
    print(f"\nAfter preprocessing - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
    
    # Save preprocessed datasets
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/liar_train.csv', index=False)
    valid_df.to_csv(f'{output_dir}/liar_valid.csv', index=False)
    test_df.to_csv(f'{output_dir}/liar_test.csv', index=False)
    
    print(f"\nSaved preprocessed LIAR datasets to {output_dir}/")
    print(f"  - liar_train.csv")
    print(f"  - liar_valid.csv")
    print(f"  - liar_test.csv")
    print("\n" + "=" * 60)
    print("LIAR preprocessing completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    preprocess_liar()

