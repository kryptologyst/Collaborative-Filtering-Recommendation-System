"""Training script for collaborative filtering models."""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization, SVDModel
from src.data.dataset import MovieLensDataset
from src.evaluation.metrics import evaluate_model, compare_models


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_and_evaluate_models(data_dir: str = "data/raw", output_dir: str = "results") -> None:
    """Train and evaluate all models.
    
    Args:
        data_dir: Directory containing dataset files
        output_dir: Directory to save results
    """
    # Set random seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    dataset = MovieLensDataset(data_dir)
    dataset.load_data()
    
    # Split data
    train_data, test_data = dataset.split_data()
    print(f"Train data: {len(train_data)} interactions")
    print(f"Test data: {len(test_data)} interactions")
    
    # Initialize models
    models = {
        'User-based CF': UserBasedCF(),
        'Item-based CF': ItemBasedCF(),
        'Matrix Factorization': MatrixFactorization(),
        'SVD': SVDModel()
    }
    
    # Train models
    print("\nTraining models...")
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(train_data)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    # Evaluate models
    print("\nEvaluating models...")
    if trained_models:
        comparison_df = compare_models(trained_models, test_data)
        
        # Save results
        results_path = os.path.join(output_dir, "model_comparison.csv")
        comparison_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # Display results
        print("\nModel Performance:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save detailed metrics for each model
        for model_name, model in trained_models.items():
            metrics = evaluate_model(model, test_data)
            metrics_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_metrics.txt")
            
            with open(metrics_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write("=" * 50 + "\n")
                f.write(str(metrics))
            
            print(f"Detailed metrics for {model_name} saved to {metrics_path}")
    
    else:
        print("No models were successfully trained.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and evaluate collaborative filtering models")
    parser.add_argument("--data_dir", default="data/raw", help="Directory containing dataset files")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    train_and_evaluate_models(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
