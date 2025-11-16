"""Evaluation metrics for recommendation systems."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from src.core import BaseModel, EvaluationMetrics


def precision_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate Precision@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0
    
    relevant_recommended = len(set(top_k) & relevant_items)
    return relevant_recommended / k


def recall_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate Recall@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score
    """
    if not relevant_items:
        return 0.0
    
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0
    
    relevant_recommended = len(set(top_k) & relevant_items)
    return relevant_recommended / len(relevant_items)


def ndcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate NDCG@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K score
    """
    if k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate MAP@K (Mean Average Precision).
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        MAP@K score
    """
    if not relevant_items or k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, item in enumerate(top_k):
        if item in relevant_items:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / len(relevant_items)


def hit_rate_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """Calculate Hit Rate@K.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Hit Rate@K score
    """
    if not relevant_items or k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    return 1.0 if set(top_k) & relevant_items else 0.0


def coverage(recommended_items_all_users: List[List[int]], all_items: Set[int]) -> float:
    """Calculate coverage - fraction of items that can be recommended.
    
    Args:
        recommended_items_all_users: List of recommendations for each user
        all_items: Set of all items in the system
        
    Returns:
        Coverage score
    """
    if not all_items:
        return 0.0
    
    recommended_items = set()
    for user_recommendations in recommended_items_all_users:
        recommended_items.update(user_recommendations)
    
    return len(recommended_items) / len(all_items)


def diversity(recommended_items_all_users: List[List[int]], item_similarity_matrix: np.ndarray, 
              item_to_idx: Dict[int, int]) -> float:
    """Calculate diversity - average pairwise dissimilarity of recommendations.
    
    Args:
        recommended_items_all_users: List of recommendations for each user
        item_similarity_matrix: Item similarity matrix
        item_to_idx: Mapping from item ID to matrix index
        
    Returns:
        Diversity score
    """
    if not recommended_items_all_users:
        return 0.0
    
    total_diversity = 0.0
    count = 0
    
    for user_recommendations in recommended_items_all_users:
        if len(user_recommendations) < 2:
            continue
            
        user_diversity = 0.0
        pair_count = 0
        
        for i in range(len(user_recommendations)):
            for j in range(i + 1, len(user_recommendations)):
                item_i = user_recommendations[i]
                item_j = user_recommendations[j]
                
                if item_i in item_to_idx and item_j in item_to_idx:
                    idx_i = item_to_idx[item_i]
                    idx_j = item_to_idx[item_j]
                    similarity = item_similarity_matrix[idx_i, idx_j]
                    user_diversity += 1.0 - similarity
                    pair_count += 1
        
        if pair_count > 0:
            total_diversity += user_diversity / pair_count
            count += 1
    
    return total_diversity / count if count > 0 else 0.0


def evaluate_model(model: BaseModel, test_data: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> EvaluationMetrics:
    """Evaluate a recommendation model.
    
    Args:
        model: Trained recommendation model
        test_data: Test dataset
        k_values: List of K values for evaluation
        
    Returns:
        EvaluationMetrics object with all scores
    """
    metrics = EvaluationMetrics()
    
    # Group test data by user
    user_test_items = defaultdict(set)
    for _, row in test_data.iterrows():
        user_test_items[row['user_id']].add(row['item_id'])
    
    # Get all items for coverage calculation
    all_items = set(test_data['item_id'].unique())
    
    # Collect metrics for each user
    precision_scores = defaultdict(list)
    recall_scores = defaultdict(list)
    ndcg_scores = defaultdict(list)
    map_scores = defaultdict(list)
    hit_rate_scores = defaultdict(list)
    all_recommendations = []
    
    for user_id, relevant_items in user_test_items.items():
        if not relevant_items:
            continue
            
        # Get recommendations
        recommendations = model.recommend(user_id, n_recommendations=max(k_values))
        recommended_items = [item_id for item_id, _ in recommendations]
        all_recommendations.append(recommended_items)
        
        # Calculate metrics for each K
        for k in k_values:
            precision_scores[k].append(precision_at_k(recommended_items, relevant_items, k))
            recall_scores[k].append(recall_at_k(recommended_items, relevant_items, k))
            ndcg_scores[k].append(ndcg_at_k(recommended_items, relevant_items, k))
            map_scores[k].append(map_at_k(recommended_items, relevant_items, k))
            hit_rate_scores[k].append(hit_rate_at_k(recommended_items, relevant_items, k))
    
    # Calculate average metrics
    for k in k_values:
        metrics.add_metric(f'precision@{k}', np.mean(precision_scores[k]))
        metrics.add_metric(f'recall@{k}', np.mean(recall_scores[k]))
        metrics.add_metric(f'ndcg@{k}', np.mean(ndcg_scores[k]))
        metrics.add_metric(f'map@{k}', np.mean(map_scores[k]))
        metrics.add_metric(f'hit_rate@{k}', np.mean(hit_rate_scores[k]))
    
    # Calculate coverage
    coverage_score = coverage(all_recommendations, all_items)
    metrics.add_metric('coverage', coverage_score)
    
    return metrics


def compare_models(models: Dict[str, BaseModel], test_data: pd.DataFrame, 
                   k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compare multiple models and return results as DataFrame.
    
    Args:
        models: Dictionary of model name -> model instance
        test_data: Test dataset
        k_values: List of K values for evaluation
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, test_data, k_values)
        
        result_row = {'model': model_name}
        result_row.update(metrics.to_dict())
        results.append(result_row)
    
    return pd.DataFrame(results)
