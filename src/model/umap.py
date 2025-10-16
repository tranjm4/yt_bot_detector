import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load feature data from CSV file.

    Args:
        data_path: Path to CSV data file

    Returns:
        pd.DataFrame: Loaded feature data
    """
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_features(df: pd.DataFrame,
                     exclude_cols: Optional[list] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare feature matrix for UMAP, handling missing values and non-numeric columns.

    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude (e.g., ['userId', 'label'])

    Returns:
        Tuple of (feature matrix, metadata dataframe)
    """
    if exclude_cols is None:
        exclude_cols = ['userId']

    # Separate metadata from features
    metadata_cols = [col for col in exclude_cols if col in df.columns]
    metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame()

    # Select numeric features only
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features_df = df[feature_cols].select_dtypes(include=[np.number])

    # Handle missing values
    features_df = features_df.fillna(features_df.median())

    print(f"Using {len(features_df.columns)} numeric features")
    print(f"Feature matrix shape: {features_df.shape}")

    return features_df.values, metadata


def fit_model(data: np.ndarray,
              n_neighbors: int = 15,
              min_dist: float = 0.1,
              n_components: int = 2,
              metric: str = "euclidean",
              random_state: int = 42) -> umap.UMAP:
    """
    Fit UMAP model to data.

    Args:
        data: Feature matrix (n_samples, n_features)
        n_neighbors: Number of neighbors for local neighborhood
        min_dist: Minimum distance between points in embedding
        n_components: Dimensionality of output embedding
        metric: Distance metric to use
        random_state: Random seed for reproducibility

    Returns:
        Fitted UMAP model
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=True
    )

    print(f"\nFitting UMAP with parameters:")
    print(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")
    print(f"  n_components={n_components}, metric={metric}")

    reducer.fit(data)
    print("UMAP fitting complete")

    return reducer


def visualize_embedding(embedding: np.ndarray,
                        metadata: Optional[pd.DataFrame] = None,
                        output_path: Optional[str] = None):
    """
    Visualize 2D UMAP embedding.

    Args:
        embedding: UMAP embedding (n_samples, 2)
        metadata: Optional metadata for coloring (e.g., labels)
        output_path: Optional path to save figure
    """
    if embedding.shape[1] != 2:
        print(f"Warning: Cannot visualize {embedding.shape[1]}D embedding. Skipping visualization.")
        return

    plt.figure(figsize=(12, 8))

    # If metadata has a 'label' column, use it for coloring
    if metadata is not None and 'label' in metadata.columns:
        labels = metadata['label']
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                            c=labels, cmap='viridis',
                            alpha=0.6, s=10)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1],
                   alpha=0.6, s=10, c='steelblue')

    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Embedding Visualization')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = ArgumentParser(description="UMAP model fitting for dimensionality reduction and clustering")
    parser.add_argument("data_path",
                        help="Path to CSV data file")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="Number of neighbors of local neighborhood; "
                             "lower values preserve locality, higher values preserve global structure")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="Determines compactness of UMAP fitting. Lower values imply tighter representations")
    parser.add_argument("--n_components", type=int, default=2,
                        help="Dimensionality of resulting UMAP projection")
    parser.add_argument("--metric", default="euclidean",
                        help="Distance metric (euclidean, cosine, manhattan, etc.)")
    parser.add_argument("--exclude_cols", nargs="+", default=["userId"],
                        help="Columns to exclude from feature matrix")
    parser.add_argument("--output_dir", default="results/umap",
                        help="Directory to save results")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_data(args.data_path)
    features, metadata = prepare_features(df, exclude_cols=args.exclude_cols)

    # Fit UMAP model
    reducer = fit_model(
        features,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=args.random_state
    )

    # Transform data
    embedding = reducer.transform(features)

    # Save results
    embedding_df = pd.DataFrame(
        embedding,
        columns=[f'umap_{i+1}' for i in range(args.n_components)]
    )

    if not metadata.empty:
        result_df = pd.concat([metadata.reset_index(drop=True), embedding_df], axis=1)
    else:
        result_df = embedding_df

    output_csv = output_dir / "umap_embedding.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"\nEmbedding saved to: {output_csv}")

    # Save model
    model_path = output_dir / "umap_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(reducer, f)
    print(f"Model saved to: {model_path}")

    # Visualize if 2D
    if args.n_components == 2:
        viz_path = output_dir / "umap_visualization.png"
        visualize_embedding(embedding, metadata, str(viz_path))


if __name__ == "__main__":
    main()