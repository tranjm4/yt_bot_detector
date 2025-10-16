"""
File: tests/test_feature_engineering.py

This file contains unit tests for various feature engineering functions
"""
import pandas as pd
import numpy as np
from datetime import datetime

import src.processing.features as features
from src.data.psql import Psql

import pytest
from unittest.mock import MagicMock, patch

import os
from dotenv import load_dotenv
load_dotenv("..")

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
    """Makes a client connection to the Postgres server"""
    client = Psql()
    
    yield client
    
    client.close_db()
    
    
class TestUserFeatures:
    
    def test_reply_ratio(self, psql_client):
        """Tests for correct computation and formatting of reply ratio function"""
        reply_ratios = features.get_reply_ratio(psql_client)
        
        assert reply_ratios["reply_ratio"].apply(
            lambda x: type(x) == np.float32
        ).all()
        
        assert reply_ratios["reply_ratio"].apply(
            lambda x: (x is not None and x is not np.nan) and 0 < x < 1).all()
        
    def test_get_comment_counts(self, psql_client):
        """Tests for correct computation and formatting of comment count function"""
        comment_counts = features.get_comment_count(psql_client, min_comments=12)
        
        assert comment_counts["comment_count"].apply(
            lambda x: x >= 12
        ).all()
        
        comment_counts = features.get_comment_count(psql_client, min_comments=5)
        assert comment_counts["comment_count"].apply(
            lambda x: x >= 5
        ).all()
        
    def test_get_user_creation_date(self, psql_client):
        """Tests for correct computation and formatting of account creation date function"""
        creation_dates = features.get_creation_dates(psql_client)
        
        assert creation_dates["account_creation_date"].apply(
            lambda x: isinstance(x, datetime)
        ).all()
    


class TestBehaviorFeatures:
    def test_circular_mean(self):
        """Tests for correct circular mean computation"""
        hours = pd.DataFrame([0, 22])
        assert features.compute_circular_mean_hours(hours) == 23
        
        hours = pd.DataFrame([23, 0])
        assert features.compute_circular_mean_hours(hours) == 23.5
    
    def test_deviance_from_mean(self):
        df = pd.Series(np.arange(0, 24))
        result_df = pd.Series([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        pd.testing.assert_series_equal(
            features.compute_hour_deviance(df, mean_hour=12),
            result_df
        )
    


class TestNewFeatures:
    """Tests for newly added feature functions"""

    def test_get_comment_latency(self, psql_client):
        """Tests comment latency feature computation"""
        latency_stats = features.get_comment_latency(psql_client, min_comments=2)

        # Check all expected columns exist
        assert set(latency_stats.columns) == {
            "userId", "mean_latency_minutes", "std_latency_minutes",
            "valid_comment_count", "filtered_comment_count"
        }

        # Skip if no data
        if len(latency_stats) == 0:
            pytest.skip("No data in test database")

        # Check that mean latencies are non-negative
        assert (latency_stats["mean_latency_minutes"] >= 0).all()

        # Check that all users have at least min_comments
        assert (latency_stats["valid_comment_count"] >= 2).all()

    def test_get_channel_diversity(self, psql_client):
        """Tests channel diversity feature computation"""
        diversity_stats = features.get_channel_diversity(psql_client, min_comments=2)

        # Check columns
        assert set(diversity_stats.columns) == {
            "userId", "unique_channel_count", "comment_count"
        }

        # Skip if no data
        if len(diversity_stats) == 0:
            pytest.skip("No data in test database")

        # Unique channels should be <= total comments
        assert (diversity_stats["unique_channel_count"] <= diversity_stats["comment_count"]).all()

        # Unique channels should be >= 1
        assert (diversity_stats["unique_channel_count"] >= 1).all()

    def test_get_text_length_stats(self, psql_client):
        """Tests text length statistics computation"""
        text_stats = features.get_text_length_stats(psql_client, min_comments=2)

        # Check columns
        assert set(text_stats.columns) == {
            "userId", "mean_text_length", "std_text_length", "comment_count"
        }

        # Skip if no data
        if len(text_stats) == 0:
            pytest.skip("No data in test database")

        # Mean text length should be positive
        assert (text_stats["meanTextLength"] > 0).all()

        # Comment count should meet minimum
        assert (text_stats["comment_count"] >= 2).all()

    def test_get_text_repetition_stats(self, psql_client):
        """Tests text repetition statistics computation"""
        repetition_stats = features.get_text_repetition_stats(psql_client, min_comments=2)

        # Check columns
        assert set(repetition_stats.columns) == {
            "userId", "unique_comment_ratio", "max_duplicate_count",
            "avg_duplicate_count", "total_duplicates", "comment_count"
        }

        # Skip if no data
        if len(repetition_stats) == 0:
            pytest.skip("No data in test database")

        # Unique comment ratio should be between 0 and 1
        assert (repetition_stats["uniqueCommentRatio"] > 0).all()
        assert (repetition_stats["uniqueCommentRatio"] <= 1).all()

        # Max duplicates should be >= avg duplicates (when avg exists)
        non_null_avg = repetition_stats[repetition_stats["avgDuplicateCount"].notna()]
        if len(non_null_avg) > 0:
            assert (non_null_avg["maxDuplicateCount"] >= non_null_avg["avgDuplicateCount"]).all()


class TestGraphFeatures:
    """Tests for graph-based features"""

    def test_build_user_video_graph(self, psql_client):
        """Tests graph construction"""
        pytest.importorskip("networkx")

        try:
            G, users, videos = features.build_user_video_graph(psql_client, sample_size=1000)

            # Check that graph has nodes
            assert len(G.nodes()) > 0
            assert len(users) > 0
            assert len(videos) > 0

            # Check that edges exist
            assert len(G.edges()) > 0
        except ValueError as e:
            # It's okay if there's not enough data
            pytest.skip(f"Not enough data for graph construction: {e}")

    def test_get_community_assignments(self, psql_client):
        """Tests community detection"""
        pytest.importorskip("networkx")

        try:
            communities = features.get_community_assignments(
                psql_client, sample_size=5000, resolution=0.5
            )

            # Check columns
            assert set(communities.columns) == {"userId", "communityId", "communitySize"}

            # Check that community IDs are non-negative integers
            assert (communities["communityId"] >= 0).all()

            # Check that community sizes are positive
            assert (communities["communitySize"] > 0).all()
        except ValueError as e:
            # It's okay if there's not enough data
            pytest.skip(f"Not enough data for community detection: {e}")

    def test_get_pagerank_scores(self, psql_client):
        """Tests PageRank computation"""
        pytest.importorskip("networkx")

        try:
            pagerank_df = features.get_pagerank_scores(psql_client, sample_size=5000)

            # Check columns
            assert set(pagerank_df.columns) == {"userId", "pagerank"}

            # Check that PageRank scores are between 0 and 1
            assert (pagerank_df["pagerank"] >= 0).all()
            assert (pagerank_df["pagerank"] <= 1).all()

            # Check that scores sum to approximately 1
            assert abs(pagerank_df["pagerank"].sum() - 1.0) < 0.01
        except ValueError as e:
            # It's okay if there's not enough data
            pytest.skip(f"Not enough data for PageRank computation: {e}")

    def test_get_temporal_co_commenting_clusters(self, psql_client):
        """Tests temporal co-commenting detection"""
        co_comment_df = features.get_temporal_co_commenting_clusters(
            psql_client, time_window_minutes=60, min_co_occurrences=2
        )

        if len(co_comment_df) == 0:
            # It's okay if no co-commenting patterns are found
            pytest.skip("No co-commenting patterns found with given parameters")

        # Check columns
        assert set(co_comment_df.columns) == {
            "userId1", "userId2", "coCommentCount", "sharedVideos"
        }

        # Check that co-comment count meets minimum
        assert (co_comment_df["coCommentCount"] >= 2).all()

        # Check that user pairs are different
        assert (co_comment_df["userId1"] != co_comment_df["userId2"]).all()