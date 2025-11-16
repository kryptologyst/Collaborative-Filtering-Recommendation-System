"""Collaborative filtering models implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.core import BaseModel


class UserBasedCF(BaseModel):
    """User-based collaborative filtering using cosine similarity."""
    
    def __init__(self, min_similarity: float = 0.1, min_common_items: int = 2):
        """Initialize user-based CF model.
        
        Args:
            min_similarity: Minimum similarity threshold for user selection
            min_common_items: Minimum number of common items required
        """
        self.min_similarity = min_similarity
        self.min_common_items = min_common_items
        self.user_similarity_matrix: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_ids: Optional[List[int]] = None
        self.item_ids: Optional[List[int]] = None
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the user-based CF model.
        
        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
        """
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Calculate user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix.values)
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
            
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # Get similar users who have rated this item
        user_ratings = self.user_item_matrix.iloc[:, item_idx]
        rated_users = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_users) < self.min_common_items:
            return 0.0
            
        # Calculate weighted average
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for rated_user in rated_users:
            if rated_user == user_id:
                continue
                
            rated_user_idx = self.user_ids.index(rated_user)
            similarity = self.user_similarity_matrix[user_idx, rated_user_idx]
            
            if similarity >= self.min_similarity:
                rating = self.user_item_matrix.iloc[rated_user_idx, item_idx]
                weighted_sum += similarity * rating
                similarity_sum += similarity
                
        return weighted_sum / similarity_sum if similarity_sum > 0 else 0.0
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find items not rated by user
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        recommendations = []
        for item_id in unrated_items:
            score = self.predict(user_id, item_id)
            if score > 0:
                recommendations.append((item_id, score))
                
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class ItemBasedCF(BaseModel):
    """Item-based collaborative filtering using cosine similarity."""
    
    def __init__(self, min_similarity: float = 0.1, min_common_users: int = 2):
        """Initialize item-based CF model.
        
        Args:
            min_similarity: Minimum similarity threshold for item selection
            min_common_users: Minimum number of common users required
        """
        self.min_similarity = min_similarity
        self.min_common_users = min_common_users
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_ids: Optional[List[int]] = None
        self.item_ids: Optional[List[int]] = None
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the item-based CF model.
        
        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
        """
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Calculate item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T.values)
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
            
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_items) < self.min_common_users:
            return 0.0
            
        # Calculate weighted average
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for rated_item in rated_items:
            rated_item_idx = self.item_ids.index(rated_item)
            similarity = self.item_similarity_matrix[item_idx, rated_item_idx]
            
            if similarity >= self.min_similarity:
                rating = self.user_item_matrix.iloc[user_idx, rated_item_idx]
                weighted_sum += similarity * rating
                similarity_sum += similarity
                
        return weighted_sum / similarity_sum if similarity_sum > 0 else 0.0
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find items not rated by user
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        recommendations = []
        for item_id in unrated_items:
            score = self.predict(user_id, item_id)
            if score > 0:
                recommendations.append((item_id, score))
                
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class MatrixFactorization(BaseModel):
    """Matrix factorization using SVD."""
    
    def __init__(self, factors: int = 50):
        """Initialize matrix factorization model.
        
        Args:
            factors: Number of latent factors
        """
        self.factors = factors
        self.model = TruncatedSVD(n_components=factors, random_state=42)
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_ids: Optional[List[int]] = None
        self.item_ids: Optional[List[int]] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the matrix factorization model.
        
        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
        """
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Fit SVD
        self.model.fit(self.user_item_matrix.values)
        
        # Get factor matrices
        self.user_factors = self.model.transform(self.user_item_matrix.values)
        self.item_factors = self.model.components_.T
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
            
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # Predict using factor matrices
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return float(prediction)
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find items not rated by user
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        recommendations = []
        for item_id in unrated_items:
            score = self.predict(user_id, item_id)
            if score > 0:
                recommendations.append((item_id, score))
                
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class SVDModel(BaseModel):
    """SVD matrix factorization using sklearn."""
    
    def __init__(self, n_factors: int = 100):
        """Initialize SVD model.
        
        Args:
            n_factors: Number of factors
        """
        self.n_factors = n_factors
        self.model = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_ids: Optional[List[int]] = None
        self.item_ids: Optional[List[int]] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the SVD model.
        
        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
        """
        # Create user-item matrix
        self.user_item_matrix = interactions.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Fit SVD
        self.model.fit(self.user_item_matrix.values)
        
        # Get factor matrices
        self.user_factors = self.model.transform(self.user_item_matrix.values)
        self.item_factors = self.model.components_.T
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
            
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # Predict using factor matrices
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return float(prediction)
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find items not rated by user
        unrated_items = user_ratings[user_ratings == 0].index.tolist()
        
        recommendations = []
        for item_id in unrated_items:
            score = self.predict(user_id, item_id)
            if score > 0:
                recommendations.append((item_id, score))
                
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
