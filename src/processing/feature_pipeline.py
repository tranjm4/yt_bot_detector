"""
File: src/processing/feature_pipeline.py

Feature engineering pipeline with preprocessing, scaling, and transformations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class InverseRobustScaler(BaseEstimator, TransformerMixin):
    """
    Inverse transform followed by RobustScaler.
    Useful for emphasizing low values (e.g., low variance as anomalous).

    Transform: x -> 1 / (x + epsilon), then RobustScaler

    Parameters:
        epsilon: Small value to avoid division by zero (default: 1e-6)
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        X_inv = 1.0 / (np.asarray(X) + self.epsilon)
        self.scaler.fit(X_inv)
        return self

    def transform(self, X):
        X_inv = 1.0 / (np.asarray(X) + self.epsilon)
        return self.scaler.transform(X_inv)

    def fit_transform(self, X, y=None):
        X_inv = 1.0 / (np.asarray(X) + self.epsilon)
        return self.scaler.fit_transform(X_inv)


class WeightedInverseRobustScaler(BaseEstimator, TransformerMixin):
    """
    Inverse transform weighted by sample size, then RobustScaler.
    Emphasizes low variance with high sample counts as more anomalous.

    Transform: x -> weight * (1 / (x + epsilon)), then RobustScaler

    Typical usage:
        - X = std_latency_minutes
        - weight = valid_comment_count (or log(valid_comment_count))

    Parameters:
        epsilon: Small value to avoid division by zero (default: 1e-6)
        weight_col_idx: Index of weight column in X (default: 1, assumes [value, weight])
        log_weight: If True, use log(weight) instead of raw weight (default: True)
    """

    def __init__(self, epsilon=1e-6, weight_col_idx=1, log_weight=True):
        self.epsilon = epsilon
        self.weight_col_idx = weight_col_idx
        self.log_weight = log_weight
        self.scaler = RobustScaler()

    def _apply_transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            raise ValueError("WeightedInverseRobustScaler requires 2D input: [value_col, weight_col]")

        values = X[:, 0]
        weights = X[:, self.weight_col_idx]

        if self.log_weight:
            weights = np.log1p(weights)  # log(1 + weight) to handle 0s

        # Weighted inverse: higher weight amplifies the inverse
        weighted_inv = weights * (1.0 / (values + self.epsilon))
        return weighted_inv.reshape(-1, 1)

    def fit(self, X, y=None):
        X_transformed = self._apply_transform(X)
        self.scaler.fit(X_transformed)
        return self

    def transform(self, X):
        X_transformed = self._apply_transform(X)
        return self.scaler.transform(X_transformed)

    def fit_transform(self, X, y=None):
        X_transformed = self._apply_transform(X)
        return self.scaler.fit_transform(X_transformed)


@dataclass
class FeatureConfig:
    """Configuration for a feature group"""
    name: str
    features: List[str]  # Feature names to compute
    params: Dict[str, Any]  # Parameters for feature computation
    log_transforms: bool = False
    scaling_method: Optional[str] = None  # 'standard', 'minmax', 'robust', or None
    impute_strategy: Optional[str] = 'median'  # 'mean', 'median', 'most_frequent', or None
    drop_if_null_pct: Optional[float] = None  # Drop feature if > this % null


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    Handles:
    - Feature computation via FeatureStore
    - Missing value imputation
    - Scaling/normalization
    - Feature selection
    - Train/test consistency

    Example:
        pipeline = FeaturePipeline(feature_store)

        # Define feature groups
        pipeline.add_feature_group(
            name='basic_features',
            features=['comment_count', 'text_length_stats'],
            scaling_method='standard'
        )

        # Fit on training data
        X_train = pipeline.fit_transform(user_ids_train)

        # Transform test data with same preprocessing
        X_test = pipeline.transform(user_ids_test)
    """

    def __init__(self, feature_store):
        """
        Initialize pipeline

        Args:
            feature_store: FeatureStore instance for computing features
        """
        self.feature_store = feature_store
        self.feature_groups: List[FeatureConfig] = []
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_columns: Optional[List[str]] = None
        self.group_to_columns: Dict[str, List[str]] = {}  # Track which columns belong to which group
        self.is_fitted = False

    def add_feature_group(self, name: str, features: List[str],
                         params: Optional[Dict] = None,
                         log_transform: Optional[bool] = False,
                         scaling_method: Optional[str] = 'standard',
                         impute_strategy: Optional[str] = 'median',
                         drop_if_null_pct: Optional[float] = 0.5):
        """
        Add a feature group to the pipeline

        Args:
            name: Name for this feature group
            features: List of feature names from FeatureRegistry
            params: Parameters to pass to feature computation
            scaling_method: 'standard', 'minmax', 'robust', or None
            impute_strategy: 'mean', 'median', 'most_frequent', or None
            drop_if_null_pct: Drop features with > this % missing (0.5 = 50%)
        """
        config = FeatureConfig(
            name=name,
            features=features,
            params=params or {},
            log_transforms=log_transform,
            scaling_method=scaling_method,
            impute_strategy=impute_strategy,
            drop_if_null_pct=drop_if_null_pct
        )
        self.feature_groups.append(config)
        logger.info(f"Added feature group '{name}' with {len(features)} features")

    def fit_transform(self, user_ids: Optional[List[str]] = None,
                     use_cache: bool = True) -> pd.DataFrame:
        """
        Fit pipeline on data and transform

        Args:
            user_ids: Optional list of user IDs to filter to
            use_cache: Use cached features if available

        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        logger.info("Fitting feature pipeline...")

        # Compute all features and track columns per group
        all_features = []
        self.group_to_columns = {}  # Reset mapping

        for group in self.feature_groups:
            logger.info(f"Computing feature group '{group.name}'...")

            # Compute features for this group
            group_dfs = []
            for feat in group.features:
                feat_df = self.feature_store.compute_feature(
                    feature_name=feat,
                    use_cache=use_cache,
                    **group.params
                )
                group_dfs.append(feat_df)

            # Merge features within this group
            group_merged = group_dfs[0]
            for feat_df in group_dfs[1:]:
                group_merged = group_merged.merge(feat_df, on='userId', how='outer', suffixes=('', '_dup'))
                group_merged = group_merged.loc[:, ~group_merged.columns.str.endswith('_dup')]

            # Track columns for this group (excluding userId)
            self.group_to_columns[group.name] = [col for col in group_merged.columns if col != 'userId']
            all_features.append(group_merged)

        # Merge all feature groups
        df = all_features[0]
        for features_df in all_features[1:]:
            df = df.merge(features_df, on='userId', how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.str.endswith('_dup')]

        # Filter to specific users if provided
        if user_ids is not None:
            df = df[df['userId'].isin(user_ids)]

        logger.info(f"Raw feature matrix: {df.shape}")
        logger.info(f"Columns in raw feature matrix: {list(df.columns)}")

        # Store userId separately
        user_id_col = df['userId'].copy()
        df = df.drop(columns=['userId'])

        # Handle missing values and scaling per group
        processed_dfs = []
        for group in self.feature_groups:
            # Get columns for this group using the tracked mapping
            group_cols = [col for col in self.group_to_columns.get(group.name, []) if col in df.columns]

            logger.info(f"Group '{group.name}' with features {group.features}, found columns: {group_cols}")

            if not group_cols:
                logger.warning(f"Group '{group.name}' has no matching columns in DataFrame")
                continue

            group_df = df[group_cols].copy()

            # Drop features with too many nulls
            if group.drop_if_null_pct is not None and len(group_df) > 0:
                null_pcts = group_df.isnull().sum() / len(group_df)
                cols_to_keep = null_pcts[null_pcts <= group.drop_if_null_pct].index
                dropped = set(group_cols) - set(cols_to_keep)
                if dropped:
                    logger.warning(f"Dropping {len(dropped)} features due to "
                                 f">{group.drop_if_null_pct*100}% nulls: {dropped}")
                group_df = group_df[cols_to_keep]

            # Skip if no columns left after filtering
            if len(group_df.columns) == 0:
                logger.warning(f"No features remaining in group '{group.name}' after filtering")
                continue

            # Impute missing values
            if group.impute_strategy and group_df.isnull().any().any():
                imputer = SimpleImputer(strategy=group.impute_strategy)
                group_df_values = imputer.fit_transform(group_df)
                group_df = pd.DataFrame(group_df_values,
                                       columns=group_df.columns,
                                       index=group_df.index)
                self.imputers[group.name] = imputer
                logger.info(f"Imputed missing values in '{group.name}' "
                          f"using {group.impute_strategy}")

            # Apply log transform if requested
            if group.log_transforms and len(group_df) > 0:
                # Use log1p to handle zeros: log(1 + x)
                group_df = np.log1p(group_df)
                logger.info(f"Applied log1p transform to '{group.name}'")

            # Scale features
            if group.scaling_method and len(group_df) > 0:
                scaler = self._get_scaler(group.scaling_method)
                group_df_values = scaler.fit_transform(group_df)
                group_df = pd.DataFrame(group_df_values,
                                       columns=group_df.columns,
                                       index=group_df.index)
                self.scalers[group.name] = scaler
                logger.info(f"Scaled '{group.name}' using {group.scaling_method}")

            processed_dfs.append(group_df)

        # Check if we have any features to return
        if not processed_dfs:
            logger.warning("No features produced after processing. Returning empty DataFrame with userId only.")
            result = pd.DataFrame({'userId': user_id_col.values})
            self.feature_columns = []
            self.is_fitted = True
            return result

        # Combine processed groups
        result = pd.concat(processed_dfs, axis=1)
        result.insert(0, 'userId', user_id_col.values)

        self.feature_columns = [col for col in result.columns if col != 'userId']
        self.is_fitted = True

        logger.info(f"Final feature matrix: {result.shape} "
                   f"({len(self.feature_columns)} features)")

        return result

    def transform(self, user_ids: Optional[List[str]] = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline

        Args:
            user_ids: Optional list of user IDs to filter to
            use_cache: Use cached features if available

        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform(). "
                             "Use fit_transform() first.")

        logger.info("Transforming data with fitted pipeline...")

        # Compute features (same as fit_transform)
        all_features = []
        for group in self.feature_groups:
            features_df = self.feature_store.compute_features(
                feature_names=group.features,
                use_cache=use_cache,
                **group.params
            )
            all_features.append(features_df)

        df = all_features[0]
        for features_df in all_features[1:]:
            df = df.merge(features_df, on='userId', how='outer', suffixes=('', '_dup'))
            df = df.loc[:, ~df.columns.str.endswith('_dup')]

        if user_ids is not None:
            df = df[df['userId'].isin(user_ids)]

        user_id_col = df['userId'].copy()
        df = df.drop(columns=['userId'])

        # Apply fitted transformations
        processed_dfs = []
        for group in self.feature_groups:
            group_cols = [col for col in df.columns
                         if any(feat in col for feat in group.features)]

            if not group_cols:
                continue

            group_df = df[group_cols].copy()

            # Apply same column filtering as fit
            if group.drop_if_null_pct is not None:
                # Only keep columns that were kept during fit
                fitted_cols = [col for col in self.feature_columns
                             if any(feat in col for feat in group.features)]
                group_df = group_df[[col for col in fitted_cols if col in group_df.columns]]

            # Skip if no columns left
            if len(group_df.columns) == 0:
                continue

            # Apply fitted imputer
            if group.name in self.imputers:
                group_df_values = self.imputers[group.name].transform(group_df)
                group_df = pd.DataFrame(group_df_values,
                                       columns=group_df.columns,
                                       index=group_df.index)

            # Apply log transform if requested (same as during fit)
            if group.log_transforms and len(group_df) > 0:
                group_df = np.log1p(group_df)

            # Apply fitted scaler
            if group.name in self.scalers:
                group_df_values = self.scalers[group.name].transform(group_df)
                group_df = pd.DataFrame(group_df_values,
                                       columns=group_df.columns,
                                       index=group_df.index)

            processed_dfs.append(group_df)

        # Check if we have any features to return
        if not processed_dfs:
            logger.warning("No features produced after processing. Returning empty DataFrame with userId only.")
            return pd.DataFrame({'userId': user_id_col.values})

        result = pd.concat(processed_dfs, axis=1)
        result.insert(0, 'userId', user_id_col.values)

        return result

    def _get_scaler(self, method: str):
        """Get scaler instance by name"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'inverse_robust': InverseRobustScaler()
        }

        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}. "
                           f"Available: {list(scalers.keys())}")

        return scalers[method]

    def get_feature_importance(self, X: pd.DataFrame,
                              y: pd.Series = None) -> pd.DataFrame:
        """
        Calculate basic feature statistics/importance

        Args:
            X: Feature matrix
            y: Optional target variable for correlation

        Returns:
            pd.DataFrame: Feature statistics
        """
        stats = []

        feature_cols = [col for col in X.columns if col != 'userId']

        for col in feature_cols:
            stat = {
                'feature': col,
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'null_count': X[col].isnull().sum()
            }

            if y is not None:
                stat['correlation_with_target'] = X[col].corr(y)

            stats.append(stat)

        return pd.DataFrame(stats).sort_values('std', ascending=False)
