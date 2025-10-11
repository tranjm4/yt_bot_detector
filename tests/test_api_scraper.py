import src.data.api as api
import pytest
from unittest.mock import MagicMock, Mock, patch

class TestApiStatusHandler:
    """Tests for the api_status_handler decorator"""
    
    @pytest.fixture
    def mock_http_error(self):
        """Factory fixture for creating HTTP errors with different status codes"""
        def _create_error(status_code: int, message: str = "Error"):
            @api.api_status_handler
            def error_function():
                mock_resp = MagicMock()
                mock_resp.status = status_code
                raise api.HttpError(resp=mock_resp, content=message.encode())
            return error_function()
        return _create_error
    
    def test_api_status_handler_success(self):
        """
        Tests the expected output of an api_status_handler
            in the case of a successful response.
        
        It should return a dict containing the status code
            and the function's return value behind a 'data' key
        """
        @api.api_status_handler
        def success_function():
            return "return_value"
        
        result = success_function()
        assert result.get("status_code", "") == 200
        assert result.get("data", "") == "return_value"

    def test_api_status_handler_error(self, mock_http_error):
        """
        Tests the expected output of an api_status_handler
        in the case of a non-200-type response
        """
        @api.api_status_handler
        def error_function():
            raise api.HttpError(resp=MagicMock(), content=b"Bad request")
        
        result = mock_http_error(400, "Bad Request")
        assert result.get("data", "") == ""
    
    def test_http_error_403(self, mock_http_error, capfd):
        result = mock_http_error(400, "Bad Request")
        
        captured = capfd.readouterr()
        assert captured.out == "HTTP Error 400: Bad Request\nInvalid video ID\n"
        
    def test_http_error_403_returns_error_code(self, mock_http_error, capfd):
        result = mock_http_error(403, "Bad Request")
        
        captured = capfd.readouterr()
        assert captured.out == "HTTP Error 403: Bad Request\nComments disabled or quota exceeded\n"
        
    def test_other_http_error(self, mock_http_error, capfd):
        result = mock_http_error(500, "?")
        
        captured = capfd.readouterr()
        assert captured.out == "HTTP Error 500: ?\n"


class TestGetChannel:
    """Tests for get_channel"""
    
    @pytest.fixture
    def mock_client(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_channel_response(self):
        """Based on https://developers.google.com/youtube/v3/docs/channels#resource"""
        return {
            "items": [{
                "id": "ID123",
                "snippet": {
                    "title": "Test Channel",
                    "publishedAt": "2020-01-01T00:00:00Z"
                },
                "contentDetails": {
                    "relatedPlaylists": {
                        "uploads": "UU123"
                    }
                },
                "statistics": {
                    "subscriberCount": "100000"
                }
            }]
        }
        
    def test_get_channel_success(self, mock_client, mock_channel_response):
        """Test successful channel retrieval"""
        mock_client.channels().list().execute.return_value = mock_channel_response
        
        result = api.get_channel(mock_client, "testhandle")
        
        assert result["status_code"] == 200
        assert result["data"] == mock_channel_response
        mock_client.channels().list.assert_called_with(
            part="snippet,contentDetails,statistics",
            forHandle="testhandle"
        )
        
    def test_get_channel_failure(self, mock_client, mock_channel_response):
        """Test unsuccessful channel retrieval"""
        mock_resp = MagicMock()
        mock_resp.status = 404
        mock_client.channels().list().execute.side_effect = api.HttpError(
            resp=mock_resp,
            content=b"Channel not found"
        )
        
        result = api.get_channel(mock_client, "nonexistent")
        
        assert result["status_code"] == 404
        assert result.get("data", "") == ""
        
    
class TestGetChannelUploads:
    """Tests for get_channel_uploads function"""
    
    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_uploads_response(self):
        """Based on https://developers.google.com/youtube/v3/docs/playlistItems/list"""
        return {
            "items": [
                {
                    "contentDetails": {"videoId": "vid1", "videoPublishedAt": "2024-01-01T00:00Z"},
                    "snippet": {"title": "Video 1"}
                },
                {
                    "contentDetails": {"videoId": "vid2", "videoPublishedAt": "2024-01-01T00:00Z"},
                    "snippet": {"title": "Video 1"}
                }
            ]
        }
        
    def test_get_channel_uploads_success(self, mock_client, mock_uploads_response):
        """Test successful uploads retrieval"""
        mock_client.playlistItems().list().execute.return_value = mock_uploads_response
        
        result = api.get_channel_uploads(mock_client, "UU123")
        
        assert result["status_code"] == 200
        assert result["data"] == mock_uploads_response
        assert len(result["data"]["items"]) == 2
        

class TestGetCommentThread:
    """Tests for get_comment_thread function"""
    
    @pytest.fixture
    def mock_client(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_comment_thread(self):
        """Based on https://developers.google.com/youtube/v3/docs/commentThreads#resource"""
        return {
            "items": [
                {
                    "id": "thread1",
                    "snippet": {
                        "videoId": "vid1",
                        "topLevelComment": {
                            "id": "comment1",
                            "snippet": {
                                "authorDisplayName": "User1",
                                "authorChannelId": {"value": "UC123"},
                                "channelId": "UC234",
                                "textOriginal": "Hello",
                                "likeCount": 1,
                                "publishedAt": "2024-01-01T00:00:00Z",
                                "updatedAt": "2024-01-01T00:00:00Z"
                            }
                        }
                    }
                }
            ]
        }
    
    def test_get_comment_thread_success(self, mock_client, mock_comment_thread):
        """Tests successful comment thread retrieval"""
        mock_client.commentThreads().list().execute.return_value = mock_comment_thread
        
        result = api.get_comment_thread(mock_client, "vid1")
        
        assert result["status_code"] == 200
        assert result["data"] == mock_comment_thread
    
    def test_get_comment_thread_disabled(self, mock_client, capfd):
        """Test comments disabled (403)"""
        mock_resp = MagicMock()
        mock_resp.status = 403
        mock_client.commentThreads().list().execute.side_effect = api.HttpError(
            resp=mock_resp,
            content=b"Comments disabled"
        )
        
        result = api.get_comment_thread(mock_client, "vid1")
        
        captured = capfd.readouterr()
        
        assert result["status_code"] == 403
        assert captured.out == "HTTP Error 403: Comments disabled\nComments disabled or quota exceeded\n"