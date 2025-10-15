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
        
        assert reply_ratios["replyRatio"].apply(
            lambda x: type(x) == np.float32
        ).all()
        
        assert reply_ratios["replyRatio"].apply(
            lambda x: (x is not None and x is not np.nan) and 0 < x < 1).all()
        
    def test_get_comment_counts(self, psql_client):
        """Tests for correct computation and formatting of comment count function"""
        comment_counts = features.get_comment_count(psql_client, min_comments=12)
        
        assert comment_counts["commentCount"].apply(
            lambda x: x >= 12
        ).all()
        
        comment_counts = features.get_comment_count(psql_client, min_comments=5)
        assert comment_counts["commentCount"].apply(
            lambda x: x >= 5
        ).all()
        
    def test_get_user_creation_date(self, psql_client):
        """Tests for correct computation and formatting of account creation date function"""
        creation_dates = features.get_creation_dates(psql_client)
        
        assert creation_dates["accountCreationDate"].apply(
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
    


class TestGraphFeatures:
    pass