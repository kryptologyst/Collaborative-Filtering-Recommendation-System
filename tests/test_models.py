"""Unit tests for the recommendation system."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization, SVDModel
from data.dataset import SyntheticDataset
from evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k, hit_rate_at_k,
    coverage, evaluate_model
)


@pytest.fixture
def sample_interactions():
    """Create sample interaction data for testing."""
    return pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
        'item_id': [101, 102, 103, 101, 104, 102, 103, 105],
        'rating': [4.0, 5.0, 3.0, 4.5, 2.0, 5.0, 4.0, 3.5],
        'timestamp': [1640995200, 1640995260, 1640995320, 1640995380, 
                     1640995440, 1640995500, 1640995560, 1640995620]
    })


@pytest.fixture
def sample_items():
    """Create sample item data for testing."""
    return pd.DataFrame({
        'item_id': [101, 102, 103, 104, 105],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genres': ['Action', 'Comedy', 'Drama', 'Horror', 'Romance'],
        'description': ['Action movie', 'Comedy movie', 'Drama movie', 'Horror movie', 'Romance movie']
    })


class TestUserBasedCF:
    """Test UserBasedCF model."""
    
    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = UserBasedCF()
        model.fit(sample_interactions)
        
        assert model.user_item_matrix is not None
        assert model.user_similarity_matrix is not None
        assert len(model.user_ids) == 3  # 3 unique users
    
    def test_predict(self, sample_interactions):
        """Test rating prediction."""
        model = UserBasedCF()
        model.fit(sample_interactions)
        
        # Test prediction for existing user-item pair
        prediction = model.predict(1, 104)
        assert isinstance(prediction, float)
        assert prediction >= 0
        
        # Test prediction for non-existing user
        prediction = model.predict(999, 101)
        assert prediction == 0.0
    
    def test_recommend(self, sample_interactions):
        """Test recommendation generation."""
        model = UserBasedCF()
        model.fit(sample_interactions)
        
        recommendations = model.recommend(1, n_recommendations=3)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Check format
        for item_id, score in recommendations:
            assert isinstance(item_id, int)
            assert isinstance(score, float)


class TestItemBasedCF:
    """Test ItemBasedCF model."""
    
    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = ItemBasedCF()
        model.fit(sample_interactions)
        
        assert model.user_item_matrix is not None
        assert model.item_similarity_matrix is not None
        assert len(model.item_ids) == 5  # 5 unique items
    
    def test_predict(self, sample_interactions):
        """Test rating prediction."""
        model = ItemBasedCF()
        model.fit(sample_interactions)
        
        prediction = model.predict(1, 104)
        assert isinstance(prediction, float)
        assert prediction >= 0
    
    def test_recommend(self, sample_interactions):
        """Test recommendation generation."""
        model = ItemBasedCF()
        model.fit(sample_interactions)
        
        recommendations = model.recommend(1, n_recommendations=3)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3


class TestMatrixFactorization:
    """Test MatrixFactorization model."""
    
    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = MatrixFactorization(factors=3)  # Use fewer factors for small dataset
        model.fit(sample_interactions)
        
        assert model.user_item_matrix is not None
        assert model.user_ids is not None
        assert model.item_ids is not None
    
    def test_predict(self, sample_interactions):
        """Test rating prediction."""
        model = MatrixFactorization(factors=3)  # Use fewer factors for small dataset
        model.fit(sample_interactions)
        
        prediction = model.predict(1, 104)
        assert isinstance(prediction, float)
    
    def test_recommend(self, sample_interactions):
        """Test recommendation generation."""
        model = MatrixFactorization(factors=3)  # Use fewer factors for small dataset
        model.fit(sample_interactions)
        
        recommendations = model.recommend(1, n_recommendations=3)
        assert isinstance(recommendations, list)


class TestSVDModel:
    """Test SVDModel."""
    
    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = SVDModel(n_factors=3)  # Use fewer factors for small dataset
        model.fit(sample_interactions)
        
        assert model.user_item_matrix is not None
        assert model.user_ids is not None
        assert model.item_ids is not None
    
    def test_predict(self, sample_interactions):
        """Test rating prediction."""
        model = SVDModel(n_factors=3)  # Use fewer factors for small dataset
        model.fit(sample_interactions)
        
        prediction = model.predict(1, 104)
        assert isinstance(prediction, float)
    
    def test_recommend(self, sample_interactions):
        """Test recommendation generation."""
        model = SVDModel(n_factors=3)  # Use fewer factors for small dataset
        model.fit(sample_interactions)
        
        recommendations = model.recommend(1, n_recommendations=3)
        assert isinstance(recommendations, list)


class TestSyntheticDataset:
    """Test SyntheticDataset."""
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        dataset = SyntheticDataset(n_users=10, n_items=5, n_interactions=20)
        interactions = dataset.generate_interactions()
        
        assert isinstance(interactions, pd.DataFrame)
        assert len(interactions) <= 20
        assert all(col in interactions.columns for col in ['user_id', 'item_id', 'rating', 'timestamp'])
    
    def test_generate_items(self):
        """Test item generation."""
        dataset = SyntheticDataset(n_items=5)
        items = dataset.generate_items()
        
        assert isinstance(items, pd.DataFrame)
        assert len(items) == 5
        assert all(col in items.columns for col in ['item_id', 'title', 'genres', 'description'])


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        precision = precision_at_k(recommended, relevant, k=3)
        assert precision == 2/3  # 2 relevant out of 3 recommended
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        recall = recall_at_k(recommended, relevant, k=3)
        assert recall == 2/3  # 2 relevant out of 3 total relevant
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        ndcg = ndcg_at_k(recommended, relevant, k=3)
        assert isinstance(ndcg, float)
        assert 0 <= ndcg <= 1
    
    def test_map_at_k(self):
        """Test MAP@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        map_score = map_at_k(recommended, relevant, k=3)
        assert isinstance(map_score, float)
        assert 0 <= map_score <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        hit_rate = hit_rate_at_k(recommended, relevant, k=3)
        assert hit_rate == 1.0  # At least one relevant item in top-3
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommended_items_all_users = [[1, 2], [2, 3], [3, 4]]
        all_items = {1, 2, 3, 4, 5}
        
        cov = coverage(recommended_items_all_users, all_items)
        assert cov == 4/5  # 4 unique items recommended out of 5 total


class TestEvaluation:
    """Test evaluation functions."""
    
    def test_evaluate_model(self, sample_interactions):
        """Test model evaluation."""
        model = UserBasedCF()
        model.fit(sample_interactions)
        
        # Create test data
        test_data = pd.DataFrame({
            'user_id': [1, 2],
            'item_id': [104, 105],
            'rating': [4.0, 3.0],
            'timestamp': [1640995700, 1640995800]
        })
        
        metrics = evaluate_model(model, test_data)
        
        assert metrics is not None
        assert 'precision@5' in metrics.metrics
        assert 'recall@5' in metrics.metrics
        assert 'ndcg@5' in metrics.metrics


if __name__ == "__main__":
    pytest.main([__file__])
