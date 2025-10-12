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
        result = mock_http_error(400, "Bad Request")
        assert result.get("data", "") == ""

    def test_http_error_400(self, mock_http_error, caplog):
        result = mock_http_error(400, "Bad Request")

        assert "HTTP Error 400: Bad Request" in caplog.text
        assert "Invalid video ID" in caplog.text

    def test_http_error_403_returns_error_code(self, mock_http_error, caplog):
        result = mock_http_error(403, "Bad Request")

        assert "HTTP Error 403: Bad Request" in caplog.text
        assert "Comments disabled or quota exceeded" in caplog.text

    def test_other_http_error(self, mock_http_error, caplog):
        result = mock_http_error(500, "?")

        assert "HTTP Error 500: ?" in caplog.text


class TestYoutubeCommentScraper:
    """Tests for YoutubeCommentScraper class"""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance with mocked client"""
        with patch('src.data.api.build') as mock_build:
            mock_client = MagicMock()
            mock_build.return_value = mock_client
            scraper = api.YoutubeCommentScraper()
            scraper.client = mock_client
            yield scraper

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
                    "contentDetails": {"videoId": "vid2", "videoPublishedAt": "2024-01-02T00:00Z"},
                    "snippet": {"title": "Video 2"}
                }
            ]
        }

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

    @pytest.fixture
    def mock_commenter_details(self):
        """Mock response for get_commenter_details"""
        return {
            "items": [{
                "id": "UC123",
                "snippet": {
                    "title": "Test User",
                    "publishedAt": "2020-01-01T00:00:00Z"
                },
                "statistics": {
                    "subscriberCount": "1000",
                    "videoCount": "50",
                    "hiddenSubscriberCount": False
                }
            }]
        }

    def test_get_channel_success(self, scraper, mock_channel_response):
        """Test successful channel retrieval"""
        scraper.client.channels().list().execute.return_value = mock_channel_response

        result = scraper.get_channel("testhandle")

        assert result["status_code"] == 200
        assert result["data"] == mock_channel_response
        scraper.client.channels().list.assert_called_with(
            part="snippet,contentDetails,statistics",
            forHandle="testhandle"
        )

    def test_get_channel_failure(self, scraper):
        """Test unsuccessful channel retrieval"""
        mock_resp = MagicMock()
        mock_resp.status = 404
        scraper.client.channels().list().execute.side_effect = api.HttpError(
            resp=mock_resp,
            content=b"Channel not found"
        )

        result = scraper.get_channel("nonexistent")

        assert result["status_code"] == 404
        assert result.get("data", "") == ""

    def test_get_channel_uploads_success(self, scraper, mock_uploads_response):
        """Test successful uploads retrieval"""
        scraper.client.playlistItems().list().execute.return_value = mock_uploads_response

        result = scraper.get_channel_uploads("UU123")

        assert result["status_code"] == 200
        assert result["data"] == mock_uploads_response
        assert len(result["data"]["items"]) == 2

    def test_get_channel_uploads_with_page_token(self, scraper, mock_uploads_response):
        """Test uploads retrieval with page token"""
        scraper.client.playlistItems().list().execute.return_value = mock_uploads_response

        result = scraper.get_channel_uploads("UU123", page_token="token123")

        assert result["status_code"] == 200
        scraper.client.playlistItems().list.assert_called_with(
            part="contentDetails,id,snippet,status",
            playlistId="UU123",
            pageToken="token123",
            maxResults=50
        )

    def test_get_comment_thread_success(self, scraper, mock_comment_thread):
        """Tests successful comment thread retrieval"""
        scraper.client.commentThreads().list().execute.return_value = mock_comment_thread

        result = scraper.get_comment_thread("vid1")

        assert result["status_code"] == 200
        assert result["data"] == mock_comment_thread

    def test_get_comment_thread_disabled(self, scraper, caplog):
        """Test comments disabled (403)"""
        mock_resp = MagicMock()
        mock_resp.status = 403
        scraper.client.commentThreads().list().execute.side_effect = api.HttpError(
            resp=mock_resp,
            content=b"Comments disabled"
        )

        result = scraper.get_comment_thread("vid1")

        assert result["status_code"] == 403
        assert "HTTP Error 403: Comments disabled" in caplog.text
        assert "Comments disabled or quota exceeded" in caplog.text

    def test_get_comment_thread_with_page_token(self, scraper, mock_comment_thread):
        """Test comment thread retrieval with page token"""
        scraper.client.commentThreads().list().execute.return_value = mock_comment_thread

        result = scraper.get_comment_thread("vid1", page_token="token123")

        assert result["status_code"] == 200
        scraper.client.commentThreads().list.assert_called_with(
            part="id,replies,snippet",
            videoId="vid1",
            maxResults=100,
            pageToken="token123"
        )

    def test_get_commenter_details_success(self, scraper, mock_commenter_details):
        """Test successful commenter details retrieval"""
        scraper.client.channels().list().execute.return_value = mock_commenter_details

        result = scraper.get_commenter_details("UC123")

        assert result["status_code"] == 200
        assert result["data"] == mock_commenter_details
        scraper.client.channels().list.assert_called_with(
            part="id,snippet,statistics",
            id="UC123"
        )

    def test_process_comment_with_account_details(self, scraper, mock_commenter_details):
        """Test process_comment with successful account details fetch"""
        comment = {
            "id": "comment1",
            "snippet": {
                "authorDisplayName": "Test User",
                "authorChannelId": {"value": "UC123"},
                "channelId": "UC234",
                "textOriginal": "Test comment",
                "likeCount": 5,
                "publishedAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            }
        }

        scraper.get_commenter_details = MagicMock(return_value={
            "status_code": 200,
            "data": mock_commenter_details
        })

        result = scraper.process_comment(
            comment=comment,
            video_id="vid1",
            video_title="Test Video",
            video_publish_date="2024-01-01T00:00:00Z",
            is_reply=False,
            head_comment_id="thread1"
        )

        assert result["author_channel_id"] == "UC123"
        assert result["author_display_name"] == "Test User"
        assert result["author_created_at"] == "2020-01-01T00:00:00Z"
        assert result["author_sub_count"] == "1000"
        assert result["author_video_count"] == "50"
        assert result["video_id"] == "vid1"
        assert result["comment_id"] == "comment1"
        assert result["text"] == "Test comment"
        assert result["is_reply"] == False
        assert result["thread_id"] == "thread1"

    def test_process_comment_without_account_details(self, scraper):
        """Test process_comment when account details fetch fails"""
        comment = {
            "id": "comment1",
            "snippet": {
                "authorDisplayName": "Test User",
                "authorChannelId": {"value": "UC123"},
                "channelId": "UC234",
                "textOriginal": "Test comment",
                "likeCount": 5,
                "publishedAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            }
        }

        scraper.get_commenter_details = MagicMock(return_value={
            "status_code": 404
        })

        result = scraper.process_comment(
            comment=comment,
            video_id="vid1",
            video_title="Test Video",
            video_publish_date="2024-01-01T00:00:00Z",
            is_reply=False,
            head_comment_id="thread1"
        )

        assert result["author_channel_id"] == "UC123"
        assert result["author_display_name"] == "Test User"
        assert "author_created_at" not in result
        assert "author_sub_count" not in result
        assert "author_video_count" not in result
