"""Core data structures and interfaces for the recommendation system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all recommendation models."""
    
    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the model on interaction data.
        
        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        pass


class Dataset:
    """Base class for recommendation datasets."""
    
    def __init__(self, interactions_path: str, items_path: Optional[str] = None):
        """Initialize dataset.
        
        Args:
            interactions_path: Path to interactions CSV file
            items_path: Optional path to items metadata CSV file
        """
        self.interactions_path = interactions_path
        self.items_path = items_path
        self.interactions: Optional[pd.DataFrame] = None
        self.items: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load dataset from files.
        
        Returns:
            Tuple of (interactions_df, items_df)
        """
        self.interactions = pd.read_csv(self.interactions_path)
        
        if self.items_path:
            self.items = pd.read_csv(self.items_path)
        else:
            self.items = None
            
        return self.interactions, self.items
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets using temporal split.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.interactions is None:
            self.load_data()
            
        # Sort by timestamp
        df_sorted = self.interactions.sort_values('timestamp')
        
        # Split chronologically
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        return train_df, test_df


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
    
    def get_metric(self, name: str) -> float:
        """Get metric value.
        
        Args:
            name: Metric name
            
        Returns:
            Metric value
        """
        return self.metrics.get(name, 0.0)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.metrics.copy()
    
    def __str__(self) -> str:
        """String representation."""
        return "\n".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
