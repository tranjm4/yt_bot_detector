"""
File: tests/test_feature_store.py

Unit tests for FeatureStore and FeaturePipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data.psql import Psql
from src.processing.feature_store import FeatureStore, FeatureRegistry, FeatureMetadata
from src.processing.feature_pipeline import FeaturePipeline

from dotenv import load_dotenv
import os

load_dotenv()

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost"))
    monkeypatch.setenv("POSTGRES_PORT", os.getenv("POSTGRES_PORT", "5678"))
    monkeypatch.setenv("POSTGRES_USER", os.getenv("POSTGRES_USER", "test_user"))
    monkeypatch.setenv("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "test_password"))
    monkeypatch.setenv("POSTGRES_DB", os.getenv("POSTGRES_DB", "test_db"))


@pytest.fixture
def psql_client():
    """Database connection"""
    client = Psql()
    yield client
    client.close_db()


@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestFeatureRegistry:
    """Tests for FeatureRegistry"""

    def test_list_features(self):
        """Test listing all features"""
        features = FeatureRegistry.list_features()
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'comment_count' in features
        assert 'comment_latency' in features

    def test_get_feature_info(self):
        """Test getting feature info"""
        info = FeatureRegistry.get_feature_info('comment_latency')
        assert info['name'] == 'comment_latency'
        assert 'description' in info
        assert 'default_params' in info
        assert 'min_comments' in info['default_params']

    def test_invalid_feature(self):
        """Test error on invalid feature"""
        with pytest.raises(ValueError, match="Unknown feature"):
            FeatureRegistry.get_feature_info('nonexistent_feature')


class TestFeatureStore:
    """Tests for FeatureStore"""

    def test_initialization(self, psql_client, temp_cache_dir):
        """Test store initialization"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        assert store.cache_dir.exists()
        assert store.metadata_dir.exists()

    def test_compute_single_feature(self, psql_client, temp_cache_dir):
        """Test computing a single feature"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        df = store.compute_feature('comment_count', min_comments=5)

        assert isinstance(df, pd.DataFrame)
        assert 'userId' in df.columns
        assert 'comment_count' in df.columns

        # Skip if no data in test database
        if len(df) == 0:
            pytest.skip("No data in test database")

    def test_caching(self, psql_client, temp_cache_dir):
        """Test that caching works"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)

        # First computation
        df1 = store.compute_feature('comment_count', use_cache=False, min_comments=5)

        if len(df1) == 0:
            pytest.skip("No data in test database")

        # Second computation should use cache
        df2 = store.compute_feature('comment_count', use_cache=True, min_comments=5)

        pd.testing.assert_frame_equal(df1, df2)

    def test_compute_multiple_features(self, psql_client, temp_cache_dir):
        """Test computing multiple features"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        df = store.compute_features(
            ['comment_count', 'comment_latency'],
            min_comments=5
        )

        assert isinstance(df, pd.DataFrame)
        assert 'userId' in df.columns
        assert 'comment_count' in df.columns
        assert 'mean_latency_minutes' in df.columns

        if len(df) == 0:
            pytest.skip("No data in test database")

    def test_feature_metadata(self, psql_client, temp_cache_dir):
        """Test metadata tracking"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        df = store.compute_feature('comment_count', min_comments=5)

        if len(df) == 0:
            pytest.skip("No data in test database")

        metadata = store.get_feature_metadata('comment_count')
        assert metadata is not None
        assert metadata.feature_name == 'comment_count'
        assert metadata.row_count > 0
        assert metadata.computation_time_seconds > 0

    def test_list_cached_features(self, psql_client, temp_cache_dir):
        """Test listing cached features"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)

        # Compute some features
        df1 = store.compute_feature('comment_count', min_comments=5)
        df2 = store.compute_feature('comment_latency', min_comments=5)

        if len(df1) == 0 or len(df2) == 0:
            pytest.skip("No data in test database")

        cached = store.list_cached_features()
        assert 'comment_count' in cached
        assert 'comment_latency' in cached

    def test_clear_cache(self, psql_client, temp_cache_dir):
        """Test clearing cache"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)

        # Compute and cache
        store.compute_feature('comment_count', min_comments=5)
        assert len(store.list_cached_features()) > 0

        # Clear specific feature
        store.clear_cache('comment_count')
        cached = store.list_cached_features()
        assert 'comment_count' not in cached

    def test_profile_features(self, psql_client, temp_cache_dir):
        """Test feature profiling"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        df = store.compute_feature('comment_count', min_comments=5)

        profile = store.profile_features(df)
        assert isinstance(profile, pd.DataFrame)
        assert 'feature' in profile.columns
        assert 'null_count' in profile.columns


class TestFeaturePipeline:
    """Tests for FeaturePipeline"""

    def test_initialization(self, psql_client, temp_cache_dir):
        """Test pipeline initialization"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        pipeline = FeaturePipeline(store)
        assert not pipeline.is_fitted

    def test_add_feature_group(self, psql_client, temp_cache_dir):
        """Test adding feature groups"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        pipeline = FeaturePipeline(store)

        pipeline.add_feature_group(
            name='basic',
            features=['comment_count'],
            scaling_method='standard'
        )

        assert len(pipeline.feature_groups) == 1
        assert pipeline.feature_groups[0].name == 'basic'

    def test_fit_transform(self, psql_client, temp_cache_dir):
        """Test fitting and transforming"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        pipeline = FeaturePipeline(store)

        pipeline.add_feature_group(
            name='basic',
            features=['comment_count'],
            params={'min_comments': 5},
            scaling_method='standard'
        )

        X = pipeline.fit_transform()

        assert isinstance(X, pd.DataFrame)
        assert 'userId' in X.columns
        assert pipeline.is_fitted

        # Skip if no data
        if len(X) == 0 or len(X.columns) == 1:  # Only userId column
            pytest.skip("No data in test database")

    def test_transform_without_fit(self, psql_client, temp_cache_dir):
        """Test that transform fails without fit"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        pipeline = FeaturePipeline(store)

        pipeline.add_feature_group(
            name='basic',
            features=['comment_count'],
            scaling_method='standard'
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            pipeline.transform()

    def test_scaling_methods(self, psql_client, temp_cache_dir):
        """Test different scaling methods"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)

        for method in ['standard', 'minmax', 'robust']:
            pipeline = FeaturePipeline(store)
            pipeline.add_feature_group(
                name='test',
                features=['comment_count'],
                params={'min_comments': 5},
                scaling_method=method
            )
            X = pipeline.fit_transform()
            assert X is not None

    def test_train_test_consistency(self, psql_client, temp_cache_dir):
        """Test that train and test get same preprocessing"""
        store = FeatureStore(psql_client, cache_dir=temp_cache_dir)
        pipeline = FeaturePipeline(store)

        pipeline.add_feature_group(
            name='basic',
            features=['comment_count'],
            params={'min_comments': 5},
            scaling_method='standard'
        )

        # Fit on "train" data
        X_train = pipeline.fit_transform()

        # Skip if no data
        if len(X_train) == 0 or len(X_train.columns) == 1:
            pytest.skip("No data in test database")

        # Get train user IDs and split
        train_ids = X_train['userId'].tolist()[:int(len(X_train) * 0.8)]
        test_ids = X_train['userId'].tolist()[int(len(X_train) * 0.8):]

        if not train_ids or not test_ids:
            pytest.skip("Not enough data for train/test split")

        # Refit on train only
        pipeline2 = FeaturePipeline(store)
        pipeline2.add_feature_group(
            name='basic',
            features=['comment_count'],
            params={'min_comments': 5},
            scaling_method='standard'
        )
        X_train_split = pipeline2.fit_transform(user_ids=train_ids)
        X_test_split = pipeline2.transform(user_ids=test_ids)

        assert len(X_train_split) > 0
        assert len(X_test_split) > 0
        # Columns should match
        assert list(X_train_split.columns) == list(X_test_split.columns)