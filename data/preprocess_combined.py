"""
Preprocessing script to combine LIAR and ISOT datasets for fake news classification.

This script:
1. Loads LIAR dataset (train.tsv, valid.tsv, test.tsv)
2. Loads ISOT dataset (True.csv, Fake.csv)
3. Standardizes text features and labels
4. Combines datasets while maintaining train/valid/test splits
5. Saves preprocessed combined datasets
"""

import pandas as pd
import os

# Column names for LIAR dataset
LIAR_COLUMN_NAMES = [
    'id',
    'label',
    'statement',
    'subject',
    'speaker',
    'job_title',
    'state_info',
    'party_affiliation',
    'barely_true_counts',
    'false_counts',
    'half_true_counts',
    'mostly_true_counts',
    'pants_on_fire_counts',
    'context'
]


def load_liar_dataset():
    """Load and preprocess LIAR dataset."""
    print("Loading LIAR dataset...")
    
    train_df = pd.read_csv('data/raw/train.tsv', sep='\t', names=LIAR_COLUMN_NAMES)
    valid_df = pd.read_csv('data/raw/valid.tsv', sep='\t', names=LIAR_COLUMN_NAMES)
    test_df = pd.read_csv('data/raw/test.tsv', sep='\t', names=LIAR_COLUMN_NAMES)
    
    # Convert labels to binary
    def create_binary_labels(df):
        label_map = {
            'true': 1,
            'mostly-true': 1,
            'half-true': 0,
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
    
    train_df = create_binary_labels(train_df)
    valid_df = create_binary_labels(valid_df)
    test_df = create_binary_labels(test_df)
    
    # Standardize column names
    train_df = train_df.rename(columns={'statement': 'text', 'label_binary': 'label'})
    valid_df = valid_df.rename(columns={'statement': 'text', 'label_binary': 'label'})
    test_df = test_df.rename(columns={'statement': 'text', 'label_binary': 'label'})
    
    # Add dataset source identifier
    train_df['dataset'] = 'LIAR'
    valid_df['dataset'] = 'LIAR'
    test_df['dataset'] = 'LIAR'
    
    # Select only relevant columns
    train_df = train_df[['text', 'label', 'dataset']]
    valid_df = valid_df[['text', 'label', 'dataset']]
    test_df = test_df[['text', 'label', 'dataset']]
    
    print(f"LIAR - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    return train_df, valid_df, test_df


def load_isot_dataset():
    """Load and preprocess ISOT dataset."""
    print("\nLoading ISOT dataset...")
    
    true_df = pd.read_csv('data/raw/True.csv')
    fake_df = pd.read_csv('data/raw/Fake.csv')
    
    # Add labels
    true_df['label'] = 1
    fake_df['label'] = 0
    
    # Combine true and fake
    isot_df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Combine title and text for better features
    isot_df['text'] = isot_df['title'].fillna('') + ' ' + isot_df['text'].fillna('')
    isot_df['text'] = isot_df['text'].str.strip()
    
    # Add dataset source identifier
    isot_df['dataset'] = 'ISOT'
    
    # Select only relevant columns
    isot_df = isot_df[['text', 'label', 'dataset']]
    
    print(f"ISOT - Total: {isot_df.shape}")
    print(f"ISOT - Label distribution:\n{isot_df['label'].value_counts()}")
    
    return isot_df


def split_isot_dataset(isot_df, train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
    """Split ISOT dataset into train/valid/test sets."""
    from sklearn.model_selection import train_test_split
    
    assert abs(train_size + valid_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    # First split: train vs (valid + test)
    train_df, temp_df = train_test_split(
        isot_df, 
        test_size=(valid_size + test_size), 
        stratify=isot_df['label'], 
        random_state=random_state
    )
    
    # Second split: valid vs test
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(test_size / (valid_size + test_size)),
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    print(f"\nISOT splits - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    return train_df, valid_df, test_df


def combine_datasets(liar_train, liar_valid, liar_test, isot_train, isot_valid, isot_test):
    """Combine LIAR and ISOT datasets."""
    print("\nCombining datasets...")
    
    combined_train = pd.concat([liar_train, isot_train], ignore_index=True)
    combined_valid = pd.concat([liar_valid, isot_valid], ignore_index=True)
    combined_test = pd.concat([liar_test, isot_test], ignore_index=True)
    
    print(f"\nCombined datasets:")
    print(f"Train: {combined_train.shape}")
    print(f"Valid: {combined_valid.shape}")
    print(f"Test: {combined_test.shape}")
    
    print(f"\nTrain label distribution:\n{combined_train['label'].value_counts()}")
    print(f"\nTrain dataset distribution:\n{combined_train['dataset'].value_counts()}")
    
    return combined_train, combined_valid, combined_test


def save_preprocessed_data(train_df, valid_df, test_df, output_dir='data/processed'):
    """Save preprocessed datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train_combined.csv', index=False)
    valid_df.to_csv(f'{output_dir}/valid_combined.csv', index=False)
    test_df.to_csv(f'{output_dir}/test_combined.csv', index=False)
    
    print(f"\nSaved preprocessed datasets to {output_dir}/")
    print(f"  - train_combined.csv")
    print(f"  - valid_combined.csv")
    print(f"  - test_combined.csv")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Combined Dataset Preprocessing")
    print("=" * 60)
    
    # Load datasets
    liar_train, liar_valid, liar_test = load_liar_dataset()
    isot_df = load_isot_dataset()
    
    # Split ISOT dataset
    isot_train, isot_valid, isot_test = split_isot_dataset(isot_df)
    
    # Combine datasets
    combined_train, combined_valid, combined_test = combine_datasets(
        liar_train, liar_valid, liar_test,
        isot_train, isot_valid, isot_test
    )
    
    # Save preprocessed data
    save_preprocessed_data(combined_train, combined_valid, combined_test)
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()


