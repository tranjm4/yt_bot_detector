import src.data.psql as psql

import pytest
from unittest.mock import MagicMock, patch

import psycopg2

from dotenv import load_dotenv
import os
import sys
from pathlib import Path

class TestPsql:
    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up test environment variables"""
        monkeypatch.setenv("POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost"))
        monkeypatch.setenv("POSTGRES_PORT", os.getenv("POSTGRES_PORT", "5678"))
        monkeypatch.setenv("POSTGRES_USER", os.getenv("POSTGRES_USER", "test_user"))
        monkeypatch.setenv("POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "test_password"))
        monkeypatch.setenv("POSTGRES_DB", os.getenv("POSTGRES_DB", "test_db"))
    
    @pytest.fixture
    def mock_channel_data(self):
        return ('UC123',)

    @pytest.fixture
    def mock_user_data(self):
        return ('user123', 'test_user', '2025-01-01 12:00:00')
    
    @pytest.fixture
    def mock_video_data(self):
        return ('vid123abc', 'Test Video', '2025-01-15 10:30:00', 'UC123')
    
    @pytest.fixture
    def mock_comment_data(self):
        return ('comment123', 'user123', 'vid123abc', False, '2025-01-16 10:00:00', None, 42, 'Test comment')

    @pytest.fixture
    def mock_psql_connection(self):
        """Mock psycopg2 connection and cursor"""
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_connection = MagicMock()

            # Mock cursor() to return mock_cursor and also support context manager
            mock_connection.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

            mock_connect.return_value = mock_connection

            yield mock_connection, mock_cursor

    def test_psql_connection_success(self):
        """Tests for a successful connection"""
        psql_client = psql.Psql()
        assert psql_client.connection is not None
        psql_client.close_db()
        
    def test_psql_connection_failure(self):
        """Tests for a failed connection"""
        with patch('psycopg2.connect') as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError("Connection failed")

            with pytest.raises(psycopg2.errors.ConnectionException) as exception_info:
                psql.Psql()

            assert "Failed to connect to PSQL database" in str(exception_info.value)
    
    def test_psql_peek_channels(self, mock_psql_connection, mock_channel_data):
        """Tests for a peek operation from the Channels table"""
        _, mock_cursor = mock_psql_connection
        mock_cursor.fetchone.return_value = mock_channel_data

        psql_client = psql.Psql()
        result = psql_client.peek("Channels")

        assert result == mock_channel_data
        
    def test_psql_peek_users(self, mock_psql_connection, mock_user_data):
        """Tests for a peek operation from the Users table"""
        _, mock_cursor = mock_psql_connection
        mock_cursor.fetchone.return_value = mock_user_data
        
        psql_client = psql.Psql()
        result = psql_client.peek("Users")
        
        assert result == mock_user_data
        
    def test_psql_peek_videos(self, mock_psql_connection, mock_user_data):
        """Tests for a peek operation from the Videos table"""
        _, mock_cursor = mock_psql_connection
        mock_cursor.fetchone.return_value = mock_user_data
        
        psql_client = psql.Psql()
        result = psql_client.peek("Videos")
        
        assert result == mock_user_data
        
    def test_psql_peek_comments(self, mock_psql_connection, mock_user_data):
        """Tests for a peek operation from the Comments table"""
        _, mock_cursor = mock_psql_connection
        mock_cursor.fetchone.return_value = mock_user_data
        
        psql_client = psql.Psql()
        result = psql_client.peek("Comments")
        
        assert result == mock_user_data