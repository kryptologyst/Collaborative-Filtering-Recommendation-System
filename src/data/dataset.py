"""Data generation and loading utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import os
from datetime import datetime, timedelta
import random

from src.core import Dataset


class SyntheticDataset(Dataset):
    """Generate synthetic recommendation dataset."""
    
    def __init__(
        self, 
        n_users: int = 1000, 
        n_items: int = 500, 
        n_interactions: int = 10000,
        rating_range: Tuple[float, float] = (1.0, 5.0),
        sparsity: float = 0.95
    ):
        """Initialize synthetic dataset generator.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            n_interactions: Number of interactions to generate
            rating_range: Range of ratings (min, max)
            sparsity: Desired sparsity level
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.rating_range = rating_range
        self.sparsity = sparsity
        
    def generate_interactions(self, random_state: int = 42) -> pd.DataFrame:
        """Generate synthetic interaction data.
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with interactions
        """
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Generate user-item pairs
        max_possible_interactions = self.n_users * self.n_items
        actual_interactions = int(max_possible_interactions * (1 - self.sparsity))
        actual_interactions = min(actual_interactions, self.n_interactions)
        
        # Generate interactions with some structure
        interactions = []
        
        # Add some popular items (power law distribution)
        item_popularity = np.random.power(0.5, self.n_items)
        item_popularity = item_popularity / item_popularity.sum()
        
        # Generate interactions
        for _ in range(actual_interactions):
            user_id = np.random.randint(1, self.n_users + 1)
            
            # Sample item based on popularity
            item_id = np.random.choice(self.n_items, p=item_popularity) + 1
            
            # Generate rating with some user bias
            user_bias = np.random.normal(0, 0.5)
            item_bias = np.random.normal(0, 0.3)
            base_rating = np.random.normal(3.0, 0.8)
            
            rating = base_rating + user_bias + item_bias
            rating = np.clip(rating, self.rating_range[0], self.rating_range[1])
            
            # Generate timestamp (last 2 years)
            days_ago = np.random.randint(0, 730)
            timestamp = int((datetime.now() - timedelta(days=days_ago)).timestamp())
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': round(rating, 1),
                'timestamp': timestamp
            })
        
        return pd.DataFrame(interactions)
    
    def generate_items(self) -> pd.DataFrame:
        """Generate synthetic item metadata.
        
        Returns:
            DataFrame with item metadata
        """
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary']
        
        items = []
        for item_id in range(1, self.n_items + 1):
            # Generate title
            title = f"Movie {item_id}"
            
            # Generate genres (1-3 genres per item)
            n_genres = np.random.randint(1, 4)
            item_genres = np.random.choice(genres, n_genres, replace=False)
            
            # Generate description
            description = f"A {', '.join(item_genres).lower()} movie about..."
            
            items.append({
                'item_id': item_id,
                'title': title,
                'genres': '|'.join(item_genres),
                'description': description
            })
        
        return pd.DataFrame(items)
    
    def save_dataset(self, output_dir: str = "data/raw", random_state: int = 42) -> None:
        """Generate and save synthetic dataset.
        
        Args:
            output_dir: Directory to save files
            random_state: Random seed for reproducibility
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data
        interactions = self.generate_interactions(random_state)
        items = self.generate_items()
        
        # Save to files
        interactions.to_csv(f"{output_dir}/interactions.csv", index=False)
        items.to_csv(f"{output_dir}/items.csv", index=False)
        
        print(f"Generated dataset with {len(interactions)} interactions and {len(items)} items")
        print(f"Saved to {output_dir}/")


class MovieLensDataset(Dataset):
    """MovieLens dataset loader with synthetic fallback."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize MovieLens dataset.
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = data_dir
        self.interactions_path = f"{data_dir}/interactions.csv"
        self.items_path = f"{data_dir}/items.csv"
        
        # Check if files exist, if not generate synthetic data
        if not os.path.exists(self.interactions_path):
            print("MovieLens data not found. Generating synthetic dataset...")
            generator = SyntheticDataset()
            generator.save_dataset(data_dir)
    
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load MovieLens dataset.
        
        Returns:
            Tuple of (interactions_df, items_df)
        """
        return super().load_data()
    
    def get_user_stats(self) -> pd.DataFrame:
        """Get user statistics.
        
        Returns:
            DataFrame with user statistics
        """
        if self.interactions is None:
            self.load_data()
        
        user_stats = self.interactions.groupby('user_id').agg({
            'item_id': 'count',
            'rating': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        user_stats.columns = ['n_items', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
        return user_stats.reset_index()
    
    def get_item_stats(self) -> pd.DataFrame:
        """Get item statistics.
        
        Returns:
            DataFrame with item statistics
        """
        if self.interactions is None:
            self.load_data()
        
        item_stats = self.interactions.groupby('item_id').agg({
            'user_id': 'count',
            'rating': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        item_stats.columns = ['n_users', 'avg_rating', 'rating_std', 'min_rating', 'max_rating']
        return item_stats.reset_index()
