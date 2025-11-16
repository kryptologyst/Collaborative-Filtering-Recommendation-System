"""Utility functions for the recommendation system."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import pickle
import json
from pathlib import Path


def save_model(model, filepath: str) -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str):
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results(results: Dict, filepath: str) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing results
        filepath: Path to save the results
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary containing results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """Create user-item matrix from interactions.
    
    Args:
        interactions: DataFrame with columns ['user_id', 'item_id', 'rating']
        
    Returns:
        User-item matrix with users as rows and items as columns
    """
    return interactions.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )


def get_popular_items(interactions: pd.DataFrame, n: int = 10) -> List[int]:
    """Get most popular items by interaction count.
    
    Args:
        interactions: DataFrame with interactions
        n: Number of popular items to return
        
    Returns:
        List of item IDs sorted by popularity
    """
    item_counts = interactions['item_id'].value_counts()
    return item_counts.head(n).index.tolist()


def get_active_users(interactions: pd.DataFrame, n: int = 10) -> List[int]:
    """Get most active users by interaction count.
    
    Args:
        interactions: DataFrame with interactions
        n: Number of active users to return
        
    Returns:
        List of user IDs sorted by activity
    """
    user_counts = interactions['user_id'].value_counts()
    return user_counts.head(n).index.tolist()


def calculate_sparsity(interactions: pd.DataFrame) -> float:
    """Calculate dataset sparsity.
    
    Args:
        interactions: DataFrame with interactions
        
    Returns:
        Sparsity value between 0 and 1
    """
    n_users = interactions['user_id'].nunique()
    n_items = interactions['item_id'].nunique()
    n_interactions = len(interactions)
    
    total_possible = n_users * n_items
    return 1 - (n_interactions / total_possible)


def get_cold_start_users(interactions: pd.DataFrame, min_interactions: int = 5) -> List[int]:
    """Get users with few interactions (cold start problem).
    
    Args:
        interactions: DataFrame with interactions
        min_interactions: Minimum number of interactions to not be considered cold start
        
    Returns:
        List of user IDs with few interactions
    """
    user_counts = interactions['user_id'].value_counts()
    return user_counts[user_counts < min_interactions].index.tolist()


def get_cold_start_items(interactions: pd.DataFrame, min_interactions: int = 5) -> List[int]:
    """Get items with few interactions (cold start problem).
    
    Args:
        interactions: DataFrame with interactions
        min_interactions: Minimum number of interactions to not be considered cold start
        
    Returns:
        List of item IDs with few interactions
    """
    item_counts = interactions['item_id'].value_counts()
    return item_counts[item_counts < min_interactions].index.tolist()


def split_data_temporal(interactions: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally (chronological split).
    
    Args:
        interactions: DataFrame with interactions
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df_sorted = interactions.sort_values('timestamp')
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df


def split_data_random(interactions: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data randomly.
    
    Args:
        interactions: DataFrame with interactions
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    return interactions.sample(frac=1-test_size, random_state=random_state), \
           interactions.drop(interactions.sample(frac=1-test_size, random_state=random_state).index)


def create_negative_samples(interactions: pd.DataFrame, n_negatives: int = 1) -> pd.DataFrame:
    """Create negative samples for implicit feedback.
    
    Args:
        interactions: DataFrame with positive interactions
        n_negatives: Number of negative samples per positive interaction
        
    Returns:
        DataFrame with negative samples
    """
    # Get all unique users and items
    users = interactions['user_id'].unique()
    items = interactions['item_id'].unique()
    
    # Create user-item pairs
    user_item_pairs = set(zip(interactions['user_id'], interactions['item_id']))
    
    negative_samples = []
    np.random.seed(42)
    
    for user in users:
        user_items = set(interactions[interactions['user_id'] == user]['item_id'])
        available_items = set(items) - user_items
        
        if len(available_items) > 0:
            n_samples = min(n_negatives, len(available_items))
            sampled_items = np.random.choice(list(available_items), n_samples, replace=False)
            
            for item in sampled_items:
                negative_samples.append({
                    'user_id': user,
                    'item_id': item,
                    'rating': 0.0,
                    'timestamp': interactions['timestamp'].max() + 1
                })
    
    return pd.DataFrame(negative_samples)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print progress bar.
    
    Args:
        current: Current progress
        total: Total items
        prefix: Prefix string
    """
    percent = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
    
    if current == total:
        print()
