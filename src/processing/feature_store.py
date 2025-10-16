"""
File: src/processing/feature_store.py

Feature Store for caching and managing feature engineering pipelines.
Provides a unified interface for computing, storing, and retrieving features.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from src.processing import features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a computed feature"""
    feature_name: str
    computed_at: datetime
    row_count: int
    column_names: List[str]
    computation_time_seconds: float
    parameters: Dict[str, Any]
    data_version: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data['computed_at'] = data['computed_at'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict):
        data['computed_at'] = datetime.fromisoformat(data['computed_at'])
        return cls(**data)


class FeatureRegistry:
    """Registry of available feature computation functions"""

    # Map feature names to computation functions
    FEATURE_FUNCTIONS: Dict[str, Callable] = {
        'comment_count': features.get_comment_count,
        'comment_latency': features.get_comment_latency,
        'comment_latency_variance_score': features.get_comment_latency_variance_score,
        'channel_diversity': features.get_channel_diversity,
        'text_length_stats': features.get_text_length_stats,
        'text_repetition_stats': features.get_text_repetition_stats,
        'comment_hour_stats': features.get_comment_hour_stats,
        'reply_ratio': features.get_reply_ratio,
        'creation_dates': features.get_creation_dates,
        # Graph features (expensive)
        'community_assignments': features.get_community_assignments,
        'pagerank_scores': features.get_pagerank_scores,
        'temporal_co_commenting': features.get_temporal_co_commenting_clusters,
    }

    # Default parameters for each feature
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'comment_count': {'min_comments': None},
        'comment_latency': {'min_comments': 2},
        'comment_latency_variance_score': {'min_comments': 2},
        'channel_diversity': {'min_comments': 2},
        'text_length_stats': {'min_comments': 2},
        'text_repetition_stats': {'min_comments': 2},
        'comment_hour_stats': {'min_comments': 2},
        'reply_ratio': {'min_comments': None},
        'creation_dates': {},
        'community_assignments': {'sample_size': 10000, 'resolution': 1.0},
        'pagerank_scores': {'sample_size': 10000, 'alpha': 0.85},
        'temporal_co_commenting': {'time_window_minutes': 60, 'min_co_occurrences': 3},
    }

    # Feature descriptions
    DESCRIPTIONS: Dict[str, str] = {
        'comment_count': 'Total number of comments per user',
        'comment_latency': 'Mean and std dev of time between video publish and comment',
        'comment_latency_variance_score': 'Weighted score emphasizing low variance with high sample size',
        'channel_diversity': 'Number of unique channels each user comments on',
        'text_length_stats': 'Mean and std dev of comment text length',
        'text_repetition_stats': 'Metrics on how often users copy-paste comments',
        'comment_hour_stats': 'Circular mean hour and hour deviance statistics using circular statistics',
        'reply_ratio': 'Ratio of replies to top-level comments',
        'creation_dates': 'User account creation dates',
        'community_assignments': 'Community detection via Louvain method',
        'pagerank_scores': 'PageRank scores indicating influence in comment network',
        'temporal_co_commenting': 'Users who repeatedly comment within similar timeframes',
    }

    @classmethod
    def list_features(cls) -> List[str]:
        """List all available features"""
        return list(cls.FEATURE_FUNCTIONS.keys())

    @classmethod
    def get_feature_info(cls, feature_name: str) -> Dict[str, Any]:
        """Get information about a feature"""
        if feature_name not in cls.FEATURE_FUNCTIONS:
            raise ValueError(f"Unknown feature: {feature_name}")

        return {
            'name': feature_name,
            'description': cls.DESCRIPTIONS.get(feature_name, 'No description'),
            'default_params': cls.DEFAULT_PARAMS.get(feature_name, {}),
            'function': cls.FEATURE_FUNCTIONS[feature_name].__name__
        }


class FeatureStore:
    """
    Central store for managing feature computation, caching, and retrieval.

    Example usage:
        store = FeatureStore(psql_client, cache_dir='./feature_cache')

        # Compute single feature
        df = store.compute_feature('comment_latency', min_comments=5)

        # Compute multiple features
        features_df = store.compute_features(['comment_count', 'text_length_stats'])

        # Load cached feature
        df = store.load_feature('comment_latency')
    """

    def __init__(self, psql_client, cache_dir: str = './feature_cache',
                 data_version: Optional[str] = None):
        """
        Initialize FeatureStore

        Args:
            psql_client: Database connection client
            cache_dir: Directory to store cached features
            data_version: Optional version string for data lineage tracking
        """
        self.psql_client = psql_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.data_version = data_version or datetime.now().strftime('%Y%m%d_%H%M%S')

        self.metadata_dir = self.cache_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)

        logger.info(f"FeatureStore initialized with cache_dir={cache_dir}, "
                   f"data_version={self.data_version}")

    def compute_feature(self, feature_name: str, use_cache: bool = True,
                       **kwargs) -> pd.DataFrame:
        """
        Compute a single feature with optional caching

        Args:
            feature_name: Name of feature to compute
            use_cache: If True, load from cache if available
            **kwargs: Parameters to pass to feature function

        Returns:
            pd.DataFrame: Computed feature data
        """
        # Check if cached version exists
        if use_cache:
            cached_df = self._load_from_cache(feature_name, kwargs)
            if cached_df is not None:
                logger.info(f"Loaded feature '{feature_name}' from cache")
                return cached_df

        # Get feature function
        if feature_name not in FeatureRegistry.FEATURE_FUNCTIONS:
            raise ValueError(f"Unknown feature: {feature_name}. "
                           f"Available: {FeatureRegistry.list_features()}")

        feature_func = FeatureRegistry.FEATURE_FUNCTIONS[feature_name]

        # Merge default params with user params
        params = FeatureRegistry.DEFAULT_PARAMS.get(feature_name, {}).copy()
        params.update(kwargs)

        # Compute feature
        logger.info(f"Computing feature '{feature_name}' with params {params}")
        start_time = datetime.now()

        try:
            df = feature_func(self.psql_client, **params)
        except Exception as e:
            logger.error(f"Failed to compute feature '{feature_name}': {e}")
            raise

        computation_time = (datetime.now() - start_time).total_seconds()

        # Create metadata
        metadata = FeatureMetadata(
            feature_name=feature_name,
            computed_at=datetime.now(),
            row_count=len(df),
            column_names=list(df.columns),
            computation_time_seconds=computation_time,
            parameters=params,
            data_version=self.data_version,
            description=FeatureRegistry.DESCRIPTIONS.get(feature_name)
        )

        # Cache result
        self._save_to_cache(feature_name, df, metadata, params)

        logger.info(f"Feature '{feature_name}' computed: {len(df)} rows, "
                   f"{computation_time:.2f}s")

        return df

    def compute_features(self, feature_names: List[str],
                        join_on: str = 'userId',
                        use_cache: bool = True,
                        **kwargs) -> pd.DataFrame:
        """
        Compute multiple features and join them into a single DataFrame

        Args:
            feature_names: List of feature names to compute
            join_on: Column to join features on (default 'userId')
            use_cache: If True, load from cache when available
            **kwargs: Parameters to pass to all feature functions

        Returns:
            pd.DataFrame: Combined feature matrix
        """
        logger.info(f"Computing {len(feature_names)} features: {feature_names}")

        dfs = []
        for feature_name in feature_names:
            df = self.compute_feature(feature_name, use_cache=use_cache, **kwargs)
            dfs.append(df)

        # Join all features
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on=join_on, how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns (e.g., commentCount appearing in multiple features)
            result = result.loc[:, ~result.columns.str.endswith('_dup')]

        logger.info(f"Combined feature matrix: {result.shape}")
        return result

    def load_feature(self, feature_name: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load a feature from cache

        Args:
            feature_name: Name of feature to load
            params: Parameters used when computing (to find correct cache file)

        Returns:
            pd.DataFrame: Cached feature data
        """
        params = params or {}
        df = self._load_from_cache(feature_name, params)
        if df is None:
            raise FileNotFoundError(f"No cached data found for feature '{feature_name}'")
        return df

    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a cached feature"""
        metadata_file = self._get_metadata_path(feature_name, {})
        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            data = json.load(f)

        return FeatureMetadata.from_dict(data)

    def list_cached_features(self) -> List[str]:
        """List all cached features"""
        cached = []
        for file in self.cache_dir.glob('*.pkl'):
            if file.stem != 'metadata':
                cached.append(file.stem.split('_params_')[0])
        return list(set(cached))

    def clear_cache(self, feature_name: Optional[str] = None):
        """
        Clear cached features

        Args:
            feature_name: If provided, only clear this feature. Otherwise clear all.
        """
        if feature_name:
            for file in self.cache_dir.glob(f'{feature_name}_*.pkl'):
                file.unlink()
                logger.info(f"Deleted cache file: {file.name}")
        else:
            for file in self.cache_dir.glob('*.pkl'):
                file.unlink()
            logger.info("Cleared all cache files")

    def _get_cache_path(self, feature_name: str, params: Dict) -> Path:
        """Generate cache file path based on feature name and params"""
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
        if param_str:
            filename = f"{feature_name}_params_{param_str}.pkl"
        else:
            filename = f"{feature_name}.pkl"
        return self.cache_dir / filename

    def _get_metadata_path(self, feature_name: str, params: Dict) -> Path:
        """Generate metadata file path"""
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
        if param_str:
            filename = f"{feature_name}_params_{param_str}.json"
        else:
            filename = f"{feature_name}.json"
        return self.metadata_dir / filename

    def _save_to_cache(self, feature_name: str, df: pd.DataFrame,
                      metadata: FeatureMetadata, params: Dict):
        """Save feature data and metadata to cache"""
        cache_path = self._get_cache_path(feature_name, params)
        metadata_path = self._get_metadata_path(feature_name, params)

        # Save DataFrame
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.debug(f"Cached feature to {cache_path.name}")

    def _load_from_cache(self, feature_name: str, params: Dict) -> Optional[pd.DataFrame]:
        """Load feature data from cache if it exists"""
        cache_path = self._get_cache_path(feature_name, params)

        if not cache_path.exists():
            return None

        with open(cache_path, 'rb') as f:
            df = pickle.load(f)

        return df

    def profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a profile report for computed features

        Args:
            df: Feature DataFrame to profile

        Returns:
            pd.DataFrame: Profile statistics
        """
        profile = []

        for col in df.columns:
            if col == 'userId':
                continue

            stats = {
                'feature': col,
                'dtype': str(df[col].dtype),
                'count': len(df[col]),
                'null_count': df[col].isnull().sum(),
                'null_pct': f"{(df[col].isnull().sum() / len(df) * 100):.2f}%"
            }

            # Numeric features
            if pd.api.types.is_numeric_dtype(df[col]):
                stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })

            profile.append(stats)

        return pd.DataFrame(profile)
