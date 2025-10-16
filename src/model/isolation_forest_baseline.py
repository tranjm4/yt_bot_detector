#!/usr/bin/env python3
"""
Isolation Forest Baseline Model for Bot Detection

Trains an Isolation Forest on processed features to identify anomalous users.
Bots are expected to be outliers with unusual patterns (low variance, high volume, etc.)

Usage:
    python src/model/isolation_forest_baseline.py data/basic_pipeline_v1_0/basic_pipeline_all.csv
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_data(data_path: str) -> pd.DataFrame:
    """Load processed feature data"""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} users with {len(df.columns)-1} features")
    print(f"Features: {list(df.columns)}")
    return df


def train_isolation_forest(X: pd.DataFrame, contamination: float = 0.1,
                           random_state: int = 42) -> IsolationForest:
    """
    Train Isolation Forest model

    Args:
        X: Feature matrix (without userId)
        contamination: Expected proportion of outliers (default 0.1 = 10% bots)
        random_state: Random seed for reproducibility

    Returns:
        Trained IsolationForest model
    """
    print(f"\nTraining Isolation Forest...")
    print(f"  Contamination: {contamination:.1%}")
    print(f"  Random state: {random_state}")

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        bootstrap=False,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X)

    print(f"✓ Model trained with {model.n_estimators} trees")
    return model


def predict_anomalies(model: IsolationForest, X: pd.DataFrame) -> tuple:
    """
    Predict anomalies and compute anomaly scores

    Returns:
        predictions: 1 for normal, -1 for anomaly
        scores: Anomaly scores (lower = more anomalous)
    """
    predictions = model.predict(X)
    scores = model.decision_function(X)

    n_anomalies = (predictions == -1).sum()
    pct_anomalies = n_anomalies / len(predictions) * 100

    print(f"\nPredictions:")
    print(f"  Normal users: {(predictions == 1).sum()}")
    print(f"  Anomalies (suspected bots): {n_anomalies} ({pct_anomalies:.1f}%)")

    return predictions, scores


def save_results(df: pd.DataFrame, predictions: np.ndarray,
                scores: np.ndarray, output_path: str):
    """Save predictions and scores"""
    results = df[['userId']].copy()
    results['is_anomaly'] = predictions == -1
    results['anomaly_score'] = scores
    results['anomaly_rank'] = results['anomaly_score'].rank()

    # Sort by most anomalous first
    results = results.sort_values('anomaly_score')

    results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    # Show top 10 most anomalous users
    print("\nTop 10 most anomalous users:")
    print(results.head(10)[['userId', 'anomaly_score', 'is_anomaly']].to_string(index=False))

    return results


def plot_anomaly_distribution(scores: np.ndarray, predictions: np.ndarray,
                              output_dir: str):
    """Plot distribution of anomaly scores"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(scores[predictions == 1], bins=50, alpha=0.6, label='Normal', color='green')
    axes[0].hist(scores[predictions == -1], bins=50, alpha=0.6, label='Anomaly', color='red')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Anomaly Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [scores[predictions == 1], scores[predictions == -1]]
    axes[1].boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Anomaly Score by Prediction')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'anomaly_score_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    plt.close()


def analyze_feature_importance(df: pd.DataFrame, predictions: np.ndarray):
    """Analyze which features differ most between normal and anomalous users"""
    feature_cols = [col for col in df.columns if col != 'userId']

    normal_stats = df[predictions == 1][feature_cols].mean()
    anomaly_stats = df[predictions == -1][feature_cols].mean()

    diff = (anomaly_stats - normal_stats).abs().sort_values(ascending=False)

    print("\nFeature differences (Anomaly vs Normal):")
    print("=" * 60)
    for feat in diff.index[:10]:
        normal_val = normal_stats[feat]
        anomaly_val = anomaly_stats[feat]
        print(f"  {feat:30s}: Normal={normal_val:8.3f}, Anomaly={anomaly_val:8.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest baseline for bot detection"
    )
    parser.add_argument(
        'data_path',
        help='Path to processed feature CSV file'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Expected proportion of outliers (default: 0.1)'
    )
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Directory to save results (default: ./results)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(args.data_path)

    # Prepare features (exclude userId and zero-variance features)
    feature_cols = [col for col in df.columns if col != 'userId']

    # Check for zero-variance features
    zero_var_cols = df[feature_cols].std() == 0
    if zero_var_cols.any():
        zero_var_features = zero_var_cols[zero_var_cols].index.tolist()
        print(f"\nWarning: Removing zero-variance features: {zero_var_features}")
        feature_cols = [col for col in feature_cols if col not in zero_var_features]

    X = df[feature_cols]
    print(f"\nUsing {len(feature_cols)} features for training")
    print(f"Features: {feature_cols}")

    # Train model
    model = train_isolation_forest(X, args.contamination, args.random_state)

    # Predict
    predictions, scores = predict_anomalies(model, X)

    # Save results
    results = save_results(df, predictions, scores,
                          output_dir / 'isolation_forest_predictions.csv')

    # Analyze
    analyze_feature_importance(df, predictions)

    # Plot
    plot_anomaly_distribution(scores, predictions, output_dir)

    print(f"\n{'='*60}")
    print("✓ Baseline Isolation Forest model complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
