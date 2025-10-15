import src.data.psql as psql

import pytest
from unittest.mock import MagicMock, patch

import psycopg2

from datetime import datetime

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
    def mock_version_data(self):
        return ('v1', datetime.strptime('2025-01-01 10:00:00', "%Y-%m-%d %H:%M:%S"), 'test description')
    
    @pytest.fixture
    def mock_channel_data(self):
        return ('UC123', "testChannel")

    @pytest.fixture
    def mock_user_data(self):
        return ('user123', 'test_user', datetime.strptime('2025-01-01 12:00:00', '%Y-%m-%d %H:%M:%S'), 10, 1, 'v1')
    
    @pytest.fixture
    def mock_video_data(self):
        return ('vid123abc', 'Test Video', datetime.strptime('2025-01-15 10:30:00', '%Y-%m-%d %H:%M:%S'), 'UC123', 'v1')
    
    @pytest.fixture
    def mock_comment_data(self):
        return ('comment123', 'user123', 'vid123abc', False, None, datetime.strptime('2025-02-15 10:30:00', '%Y-%m-%d %H:%M:%S'), None, 42, 'Test comment', 'v1')
    
    @pytest.fixture
    def mock_version_insert(self):
        return {
            "versionName": "v1",
            "createdAt": "2025-01-01 10:00:00",
            "versionDescription": "test description"
        }
    
    @pytest.fixture
    def mock_channel_insert(self):
        return {
            "channelId": "UC123",
            "channelName": "testChannel"
        }
        
    @pytest.fixture
    def mock_user_insert(self):
        return {
            "userId": "user123",
            "username": "test_user",
            "createDate": "2025-01-01 12:00:00",
            "subCount": 10,
            "videoCount": 1,
            "versionName": "v1"
        }
        
    @pytest.fixture
    def mock_video_insert(self):
        return {
            "videoId": "vid123abc",
            "title": "Test Video",
            "publishDate": "2025-01-15 10:30:00",
            "channelId": "UC123",
            "versionName": "v1"
        }
        
    @pytest.fixture
    def mock_comment_insert(self):
        return {
            "commentId": "comment123",
            "commenterId": "user123",
            "videoId": "vid123abc",
            "isReply": False,
            "threadId": None,
            "publishDate": "2025-02-15 10:30:00",
            "editDate": None,
            "likeCount": 42,
            "commentText": "Test comment",
            "versionName": "v1"
        }

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
            
    @pytest.fixture
    def clean_db(self):
        """Cleans database before and after each test"""
        psql_client = psql.Psql()
        cursor = psql_client.connection.cursor()

        try:
            # Clean before test
            cursor.execute("TRUNCATE TABLE Yt.Comments, Yt.Videos, Yt.Users, Yt.Channels, Yt.Versions CASCADE;")
            psql_client.connection.commit()
        except Exception:
            psql_client.connection.rollback()
            raise

        yield psql_client

        try:
            # Clean up after test
            cursor.execute("TRUNCATE TABLE Yt.Comments, Yt.Videos, Yt.Users, Yt.Channels, Yt.Versions CASCADE;")
            psql_client.connection.commit()
        except Exception:
            psql_client.connection.rollback()

        cursor.close()
        psql_client.close_db()
        
    @pytest.fixture
    def inserted_comment(self, clean_db, 
                        mock_version_insert,
                        mock_channel_insert, 
                        mock_user_insert,
                        mock_video_insert,
                        mock_comment_insert):
        """Fixture that inserts a complete comment with all dependencies"""
        psql_client = clean_db
        # Insert version
        version_base_model = psql.VersionFields(**mock_version_insert)
        psql_client.insert("Versions", version_base_model)
        # Insert channel
        channel_base_model = psql.ChannelFields(**mock_channel_insert)
        psql_client.insert("Channels", channel_base_model)
        # Insert user
        user_base_model = psql.UserFields(**mock_user_insert)
        psql_client.insert("Users", user_base_model)
        # Insert video
        video_base_model = psql.VideoFields(**mock_video_insert)
        psql_client.insert("Videos", video_base_model)
        # Insert comment
        comment_base_model = psql.CommentFields(**mock_comment_insert)
        psql_client.insert("Comments", comment_base_model)
        
        return psql_client

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
        
    def test_psql_peek_versions(self, mock_psql_connection, mock_version_data):
        """Tests for a peek operation from the Versions table"""
        _, mock_cursor = mock_psql_connection
        mock_cursor.fetchone.return_value = mock_version_data
        
        psql_client = psql.Psql()
        result = psql_client.peek("Versions")
        
        assert result == mock_version_data
        
    def test_psql_peek_injection_attempt(self, mock_psql_connection, mock_user_data):
        """Tests for a SQL injection peek using a different table name"""
        _, mock_cursor = mock_psql_connection
        psql_client = psql.Psql()

        with pytest.raises(ValueError) as exception_info:
            psql_client.peek("Channels; DROP TABLE Yt.Users;")

        assert "Invalid table name" in str(exception_info.value)
        
    def test_psql_channels_insertion_success(self, clean_db, mock_channel_insert, mock_channel_data):
        """Tests for a successful SQL insertion into the Channels table"""
        psql_client = clean_db
        channel_base_model = psql.ChannelFields(**mock_channel_insert)
        result = psql_client.insert("Channels", channel_base_model)
        
        assert result == True
        
        assert psql_client.peek("Channels") == mock_channel_data
        
    def test_psql_users_insertion_success(self, clean_db, mock_version_insert, mock_user_insert, mock_user_data):
        """Tests for a successful SQL insertion into the Users table"""
        psql_client = clean_db

        # Insert version first (foreign key requirement)
        version_base_model = psql.VersionFields(**mock_version_insert)
        psql_client.insert("Versions", version_base_model)

        user_base_model = psql.UserFields(**mock_user_insert)
        result = psql_client.insert("Users", user_base_model)

        assert result == True

        # Check only the first 6 fields (excluding updatedAt)
        peek_result = psql_client.peek("Users")
        assert peek_result[:6] == mock_user_data
        
    def test_psql_videos_insertion_success(self, clean_db, mock_version_insert, mock_channel_insert, mock_video_insert, mock_video_data):
        """Tests for a successful SQL insertion into the videos table"""
        psql_client = clean_db

        # Insert version
        version_base_model = psql.VersionFields(**mock_version_insert)
        psql_client.insert("Versions", version_base_model)

        # Insert channel
        channel_base_model = psql.ChannelFields(**mock_channel_insert)
        psql_client.insert("Channels", channel_base_model)

        # Insert video
        video_base_model = psql.VideoFields(**mock_video_insert)
        result = psql_client.insert("Videos", video_base_model)

        assert result == True

        # Check only the first 5 fields (excluding updatedAt)
        peek_result = psql_client.peek("Videos")
        assert peek_result[:5] == mock_video_data
        
    def test_psql_comments_insertion_success(self, clean_db,
                                             mock_channel_insert,
                                             mock_video_insert,
                                             mock_user_insert,
                                             mock_comment_insert,
                                             mock_comment_data,
                                             mock_version_insert):
        """Tests for a successful SQL isnertion into the Comments table"""
        psql_client = clean_db

        # Insert version
        version_base_model = psql.VersionFields(**mock_version_insert)
        psql_client.insert("Versions", version_base_model)

        # Insert channel
        channel_base_model = psql.ChannelFields(**mock_channel_insert)
        psql_client.insert("Channels", channel_base_model)

        # Insert user
        user_base_model = psql.UserFields(**mock_user_insert)
        psql_client.insert("Users", user_base_model)

        # Insert video
        video_base_model = psql.VideoFields(**mock_video_insert)
        psql_client.insert("Videos", video_base_model)

        # Insert comment
        comments_base_model = psql.CommentFields(**mock_comment_insert)
        result = psql_client.insert("Comments", comments_base_model)

        assert result == True
        # Check only the first 10 fields (excluding updatedAt)
        peek_result = psql_client.peek("Comments")
        assert peek_result[:10] == mock_comment_data
        
    def test_psql_version_insertion_success(self, clean_db, mock_version_insert, mock_version_data):
        """Tests for a successful SQL insertion into the Versions table"""
        psql_client = clean_db
        
        version_base_model = psql.VersionFields(**mock_version_insert)
        result = psql_client.insert("Versions", version_base_model)
        
        assert result == True
        assert psql_client.peek("Versions")[1:] == mock_version_data[1:] # Ignore autoincrement from previous tests
        