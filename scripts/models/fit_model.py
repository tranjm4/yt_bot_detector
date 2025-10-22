"""
File: scripts/models/run_models.py

This script handles model fitting and evaluation given various configurations.
The standard approach is as follows:
    1. Fit an isolation forest on the data
    2. Verify consistency with UMAP dimensionality reduction
    3. Verify stability across multiple isolation forest initializations
    4. Verify consistency across multiple methods
    5. Evaluates differences in features between anomalies and non-anomalies
"""

from argparse import ArgumentParser
import yaml
import os
from pathlib import Path

from typing import Dict, Literal, Optional, List
from typing_extensions import TypedDict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

import src.model.isolation_forest_baseline as iso_forest
import src.model.umap as um
from src.eval.features_eval import FeatureEvaluation

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Metadata(TypedDict):
    name: str
    version: str
    description: str
    
class UmapConfig(TypedDict):
    n_neighbors: int
    min_dist: float
    metric: Literal["euclidean", "manhattan", "chebyshev", "minkowsky"]
    n_jobs: Optional[int]
    low_memory: Optional[bool]
    subsample_fit: Optional[int]
    
class ModelConfig(TypedDict):
    data: str
    outfile_dir: str
    random_seed: int
    ignore_cols: Optional[List[str]]
    umap: UmapConfig
    
class Config(TypedDict):
    model_type: Literal["isolation_forest"] # TODO: incorporate other methods later (K-means, DBSCAN, hierarchical)
    metadata: Metadata
    config: ModelConfig
        

def read_config(path: str) -> Config:
    """
    Given the path to the config file, reads it according to the
    predefined TypedDict classes above (starting at Config)
    """
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            
        # Validate config file
        # Guarantees fields and field types
        assert config.get("model_type", "")
        assert config.get("metadata", "")
        assert config["metadata"].get("name", "")
        assert config["metadata"].get("version", "")
        assert config["metadata"].get("description", "")
        assert config.get("config", "")
        assert config["config"].get("data", "")
        assert config["config"].get("outfile_dir", "")
        assert config["config"].get("umap", "")
        assert config["config"]["umap"].get("n_neighbors", "")
        assert config["config"]["umap"].get("min_dist", "")
        assert config["config"]["umap"].get("metric", "")
        # Type checking config
        if config["config"].get("random_seed") is not None:
            assert isinstance(config["config"]["random_seed"], int)
        assert isinstance(config["config"]["umap"]["n_neighbors"], int)
        assert isinstance(config["config"]["umap"]["min_dist"], float)
    except:
        logger.error(f"Unable to read from path {path}")
        raise
    
    return Config(**config)

def _log_config(config: Config):
    """
    Verbose logging of config file
    """
    logger.info("="*50)
    logger.info("")
    logger.info(f"{'MODEL TYPE:':<15} {config['model_type']}")
    logger.info(f"{'NAME:':<15} {config['metadata']['name']}")
    logger.info("")
    logger.info(f"{'MODEL CONFIG:'}")
    logger.info(f"\t{'RANDOM SEED:':<15} {config['config']['random_seed']}")
    logger.info(f"\t{'UMAP:':<15} {config['config']['umap']}")
    logger.info("")
    logger.info("="*50)

def load_data(path: str) -> pd.DataFrame:
    try:
        with open(path, "r") as f:
            df = pd.read_csv(f)
            return df
    except:
        logger.error(f"Unable to read data path {path}")
        raise

def visualize_umap_with_labels(embedding_df: pd.DataFrame, outfile_dir: Path):
    """
    Visualize UMAP embedding colored by Isolation Forest labels.

    Args:
        embedding_df: DataFrame with UMAP coordinates and IF labels
        outfile_dir: Directory to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by IF label (anomaly vs normal)
    anomaly_mask = embedding_df['if_label'] == -1
    normal_mask = embedding_df['if_label'] == 1

    axes[0].scatter(
        embedding_df.loc[normal_mask, 'umap_1'],
        embedding_df.loc[normal_mask, 'umap_2'],
        c='steelblue', label='Normal', alpha=0.25, s=20
    )
    axes[0].scatter(
        embedding_df.loc[anomaly_mask, 'umap_1'],
        embedding_df.loc[anomaly_mask, 'umap_2'],
        c='red', label='Anomaly', alpha=0.15, s=20
    )
    axes[0].set_xlabel('UMAP Component 1')
    axes[0].set_ylabel('UMAP Component 2')
    axes[0].set_title('UMAP Embedding - Isolation Forest Labels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Color by anomaly score (continuous)
    scatter = axes[1].scatter(
        embedding_df['umap_1'],
        embedding_df['umap_2'],
        c=embedding_df['anomaly_score'],
        cmap='RdYlGn',  # Red for low (anomalous), green for high (normal)
        alpha=0.15, s=20
    )
    plt.colorbar(scatter, ax=axes[1], label='Anomaly Score')
    axes[1].set_xlabel('UMAP Component 1')
    axes[1].set_ylabel('UMAP Component 2')
    axes[1].set_title('UMAP Embedding - Anomaly Scores')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization
    viz_path = outfile_dir / "umap_if_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {viz_path}")
    plt.close()
    
def train_isolation_forest(config: ModelConfig) -> tuple[iso_forest.IsolationForest, pd.DataFrame, np.ndarray]:
    # Load data
    data_path = config["data"]
    df = load_data(data_path)

    # Ignore columns for training
    if config.get("ignore_cols", []):
        X = df.drop(labels=config.get("ignore_cols"), axis=1)
    else:
        X = df
    if config.get("random_seed"):
        model = iso_forest.train_isolation_forest(X, random_state=config["random_seed"])
    else:
        model = iso_forest.train_isolation_forest(X)

    # Run labeling for isolation forest
    labels = model.predict(X)
    df["labels"] = labels

    # Get anomaly scores for later analysis
    scores = model.decision_function(X)
    df["anomaly_score"] = scores

    logger.info(f"Isolation Forest trained. Found {(labels == -1).sum()} anomalies out of {len(labels)} samples")

    return model, df, X

def fit_umap(df: pd.DataFrame, X: np.ndarray, config: ModelConfig):
    """
    Fit UMAP on the feature data and visualize with isolation forest labels.

    Args:
        df: DataFrame with labels and anomaly scores
        X: Feature matrix used for training
        config: Model configuration containing UMAP parameters
    """
    logger.info("Fitting UMAP model...")

    # Extract UMAP configuration
    umap_config = config["umap"]

    # Get optional performance parameters
    n_jobs = umap_config.get("n_jobs", 1)
    low_memory = umap_config.get("low_memory", False)
    subsample_fit = umap_config.get("subsample_fit", None)

    # Handle parallelism vs reproducibility tradeoff
    # UMAP cannot use both random_state and n_jobs > 1
    if n_jobs and n_jobs != 1:
        logger.warning(f"Using n_jobs={n_jobs} for parallelism - disabling random_state for speed")
        random_state = None
    else:
        random_state = config.get("random_seed", 42)
        n_jobs = 1  # Force to 1 if using random_state

    # Prepare data
    X_array = X.values if isinstance(X, pd.DataFrame) else X

    # Subsample for fitting if requested (much faster for large datasets)
    if subsample_fit and subsample_fit < len(X_array):
        logger.info(f"Subsampling {subsample_fit} points from {len(X_array)} for UMAP fitting...")
        indices = np.random.RandomState(42).choice(len(X_array), size=subsample_fit, replace=False)
        X_fit = X_array[indices]
    else:
        X_fit = X_array

    # Fit UMAP model
    reducer = um.fit_model(
        data=X_fit,
        n_neighbors=umap_config["n_neighbors"],
        min_dist=umap_config["min_dist"],
        n_components=2,
        metric=umap_config["metric"],
        random_state=random_state,
        low_memory=low_memory,
        n_jobs=n_jobs or 1
    )

    # Transform ALL data (even if we only fit on subset)
    logger.info(f"Transforming all {len(X_array)} samples...")
    embedding = reducer.transform(X_array)

    # Create output directory
    outfile_dir = Path(config["outfile_dir"])
    outfile_dir.mkdir(parents=True, exist_ok=True)

    # Save UMAP embedding with labels
    embedding_df = pd.DataFrame(
        embedding,
        columns=['umap_1', 'umap_2']
    )

    # Add isolation forest labels and scores
    embedding_df['if_label'] = df['labels'].values
    embedding_df['anomaly_score'] = df['anomaly_score'].values

    # Add userId if available
    if 'userId' in df.columns:
        embedding_df['userId'] = df['userId'].values

    # Save to CSV
    output_csv = outfile_dir / "umap_with_if_labels.csv"
    embedding_df.to_csv(output_csv, index=False)
    logger.info(f"UMAP embedding saved to: {output_csv}")

    # Create visualization
    visualize_umap_with_labels(embedding_df, outfile_dir)
    iso_forest.plot_anomaly_distribution(df["anomaly_score"], predictions=df["labels"], output_dir=outfile_dir)

    # Save UMAP model
    umap_model_path = outfile_dir / "umap_model.joblib"
    joblib.dump(reducer, umap_model_path)
    logger.info(f"UMAP model saved to: {umap_model_path}")

    return reducer, embedding_df

def run_feature_evaluations(config: ModelConfig, outfile_dir: str):
    """
    Runs various feature evaluations (see src/eval/features_eval.py)
    """
    data_path = config["data"]
    df = load_data(data_path)

    evaluation = FeatureEvaluation(data_path=data_path, outdir=outfile_dir, df=df)
    evaluation.evaluate()

    return

def main():
    parser = ArgumentParser(description="Script for running and visualizing \
                            model fittings (e.g., UMAP, isolation forests)")
    parser.add_argument("config", help="File path for runtime configuration")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbosity of logging; default=False")
    args = parser.parse_args()
    config = read_config(args.config)
    logger.info("Successfully parsed config file")
    if args.verbose:
        _log_config(config)

    # Run Isolation Forest model based on config
    logger.info("=" * 60)
    logger.info("STEP 1: Training Isolation Forest")
    logger.info("=" * 60)
    if_model, df_with_labels, X = train_isolation_forest(config["config"])

    # Save Isolation Forest results
    outfile_dir = Path(config["config"]["outfile_dir"])
    outfile_dir.mkdir(parents=True, exist_ok=True)
    if_results_path = outfile_dir / "isolation_forest_predictions.csv"
    df_with_labels.to_csv(if_results_path, index=False)
    logger.info(f"Isolation Forest predictions saved to: {if_results_path}")

    # Save Isolation Forest model
    if_model_path = outfile_dir / "isolation_forest_model.joblib"
    joblib.dump(if_model, if_model_path)
    logger.info(f"Isolation Forest model saved to: {if_model_path}")

    # Run UMAP model with isolation forest as labels
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Fitting UMAP with Isolation Forest labels")
    logger.info("=" * 60)
    umap_model, umap_embedding = fit_umap(df_with_labels, X, config["config"])
    
    # Run Feature Evaluations
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Running Feature Evaluations")
    logger.info("=" * 60)
    run_feature_evaluations(config["config"], outfile_dir)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Results saved to: {outfile_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()