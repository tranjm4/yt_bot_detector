"""
File: src/data/api.py

This file details modules designed to scrape data from the Youtube API
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

import os

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from datetime import datetime

from functools import wraps


class CommentData(TypedDict):
    # Channels Table
    video_channel_id: str
    # Users Table
    author_channel_id: str
    author_display_name: str
    author_created_at: str | datetime
    author_sub_count: int
    author_video_count: int
    # Videos Table
    video_id: str
    video_title: str
    video_publish_date: str | datetime
    
    # Comments Table
    comment_id: str
    is_reply: bool
    thread_id: str
    published_at: str | datetime
    updated_at: str | datetime
    like_count: int
    text: str

CHANNEL_HANDLES = [
    "msnbc",
    "cnn",
    "FoxNews",
    "briantylercohen",
    "DailyWirePlus",
    "ABCNews",
    "TuckerCarlson"
]

API_KEY = os.getenv("GOOGLE_API_KEY")
API_VERSION = os.getenv("GOOGLE_API_VERSION")
API_SERVICE_NAME = os.getenv("GOOGLE_API_YT_SERVICE_NAME")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def api_status_handler(f: callable):
    """
    Wrapper for functions that make requests to the Youtube API.
    
    Handles status code; if there is no exception,
        the response is wrapped into the following object:
        {
            status_code: 200,
            data: response
        }
    
    Otherwise, it returns the following object:
        {
            status_code: error_status number (e.g., 400, 403, 404)
        }
        
    This is used to encourage error handling in functions that utilize
    functions wrapped by this wrapper.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return {
                "status_code": 200,
                "data": f(*args, **kwargs)
            }
        except HttpError as e:
            status_code = e.resp.status
            error_content = e.content.decode("utf-8")
            logger.error(f"HTTP Error {status_code}: {error_content}")
            
            if status_code == 403:
                logger.error("Comments disabled or quota exceeded")
            elif status_code == 404:
                logger.error("Video not found")
            elif status_code == 400:
                logger.error("Invalid video ID")
            
            return {
                "status_code": status_code
            }
        
    return wrapper

class YoutubeCommentScraper:
    def __init__(self):
        # Initializes client
        self.client = build(
            serviceName = API_SERVICE_NAME,
            version = API_VERSION,
            developerKey = API_KEY
        )

    @api_status_handler
    def get_channel(self, handle: str):
        """
        Retrieves the Channel Resource given the handle
        (https://developers.google.com/youtube/v3/docs/channels/list)
        """
        # Create request via the Google API (This is a quite convenient abstraction)
        request = self.client.channels().list(
            part="snippet,contentDetails,statistics",
            forHandle=handle
        )

        channel_response = request.execute()
        return channel_response

    @api_status_handler
    def get_channel_uploads(self, uploads_playlist_id: str, page_token: Optional[str] = None):
        """
        Retrieves the uploads playlist given the uploads ID
        (https://developers.google.com/youtube/v3/docs/playlistItems/list)
        
        Args:
            uploads_playlist_id (str): The ID to the uploads playlist to retrieve the videos
            page_token (Optional[str]): The page token used to get the next page of videos
        """
        if page_token:
            request = self.client.playlistItems().list(
                part="contentDetails,id,snippet,status",
                playlistId=uploads_playlist_id,
                pageToken=page_token,
                maxResults=20
            )
        else:
            request = self.client.playlistItems().list(
                part="contentDetails,id,snippet,status",
                playlistId=uploads_playlist_id,
                maxResults=20
            )

        uploads_response = request.execute()
        return uploads_response

    @api_status_handler
    def get_comment_thread(self, video_id: str, page_token: Optional[str] = None):
        """
        Retrieves the comment thread given the video ID
        (https://developers.google.com/youtube/v3/docs/commentThreads/list)
        
        Args:
            video_id (str): The video ID to retrieve its comments
            page_token (Optional[str]): The page token used to get the next page of videos
        """
        # Get the comment thread
        if page_token:
            request = self.client.commentThreads().list(
                part="id,replies,snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token
            )
        else:
            request = self.client.commentThreads().list(
                part="id,replies,snippet",
                videoId=video_id,
                maxResults=100
            )
            
        comment_thread_response = request.execute()
        return comment_thread_response

    @api_status_handler
    def get_commenter_details(self, account_id: str | List[str]):
        """
        Retrieves commenter details given a single account id or list of account ids
        """
        # Join multiple account ids into CSV if applicable
        id_param = account_id if isinstance(account_id, str) else ",".join(account_id)
        
        request = self.client.channels().list(
            part="id,snippet,statistics",
            id=id_param,
            maxResults=50
        )
        response = request.execute()
        
        return response

    def scrape(self) -> List[CommentData]:
        """
        Retrieves comments from the 50 most recent videos from all included channels.
        
        Returns:
            List[CommentData]: A list of CommentData TypedDicts
        """
        client = build(API_SERVICE_NAME, API_VERSION, developerKey=API_KEY)
        
        all_data = []
        for channel_handle in CHANNEL_HANDLES:
            logger.info(f"="*80)
            logger.info(f"\tRetrieving comments from: {channel_handle:^30}")
            logger.info(f"="*80)
            
            # Get channel info
            channel_response = self.get_channel(channel_handle)
            if channel_response["status_code"] == 200:
                channel = channel_response["data"]
            else:
                logger.error(f"Unable to retrieve channel data; skipping...")
                continue
            
            # Retrieve uploads ID and get videos
            playlist_id = channel["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            page_token = None
            for _ in range(1): # Limit to 20 videos per channel (1 page × 20 videos)
                uploads_response = self.get_channel_uploads(playlist_id, page_token)                      
                if uploads_response["status_code"] == 200:
                    uploads = uploads_response["data"]["items"]
                else:
                    logger.error(f"Unable to process {channel_handle} uploads playlist; skipping...")
                    continue
                
                
                comments = self.process_videos(uploads)
                all_data.extend(comments)
                
                # Update page token
                page_token = uploads_response["data"].get("nextPageToken", "")
                if not page_token:
                    break
            
        return all_data
            

    def process_videos(self, uploads: List[Dict[str, Any]]) -> List[CommentData]:
        """
        Helper function for parsing videos from a list of uploads from a channel.
        
        Args:
            uploads (List[Dict]): A list of PlaylistItem Resources
                (https://developers.google.com/youtube/v3/docs/playlistItems#resource)
        
        Returns:
            List[CommentData]: A list of all retrieved comments
        """
        comments = []
        for video_entry in uploads:
            # Get an entry's video ID
            video_id = video_entry["contentDetails"]["videoId"]
            video_title = video_entry["snippet"]["title"]
            video_publish_date = video_entry["contentDetails"]["videoPublishedAt"]
            
            logger.info(f"Retrieving from video: [{video_title}]")
            
            # Limit to 200 comments per video
            page_token = None
            for _ in range(2):
                # Use video ID for comment thread
                comment_thread_response = self.get_comment_thread(video_id, page_token)
                if comment_thread_response["status_code"] == 200:
                    comment_thread = comment_thread_response["data"]["items"]
                else:
                    logger.info(f"Video [{video_id}] has comments disabled. Skipping...")
                    continue
                
                # Retrieve all comments from video's comment thread
                video_comments = self.process_comment_thread(comment_thread,
                                                        video_id,
                                                        video_title,
                                                        video_publish_date)
                comments.extend(video_comments)

                # Update page token for next iteration
                page_token = comment_thread_response["data"].get("nextPageToken", "")
                if not page_token:
                    break
            
        return comments


    def process_comment_thread(self, comment_thread: List[Dict[str, Any]],
                            video_id: str, video_title: str,
                            video_publish_date: str | datetime
                            ) -> List[CommentData]:
        """
        Helper function for parsing comments from a single video's comment thread.
        Retrieves head comments and their replies, if any.
        
        Args:
            comment_thread (List[Dict]): A list of comments from the comment thread
            
        Returns:
            List[CommentData]: A list of retrieved comments
        """
        comments = []
        
        batch = []
        BATCH_SIZE = 50
        for head_comment in comment_thread:
            # Get video ID
            video_id = head_comment["snippet"]["videoId"]
            
            # Get head comment's comment data
            comment = head_comment["snippet"]["topLevelComment"]
            comment = {**comment, "videoId": video_id, "videoTitle": video_title, "videoPublishDate": video_publish_date}
            batch.append(comment)
            if len(batch) >= BATCH_SIZE:
                batch_comments = self.batch_process_comments(batch)
                comments.extend(batch_comments)
                batch = []
            
            # If the head comment has replies, get those comments
            replies = head_comment.get("replies", {})
            thread_replies = replies.get("comments", [])
            for reply in thread_replies:
                reply = {**reply, "videoId": video_id, "videoTitle": video_title, "videoPublishDate": video_publish_date}
                batch.append(reply)
                if len(batch) >= BATCH_SIZE:
                    batch_comments = self.batch_process_comments(batch)
                    comments.extend(batch_comments)
                    batch = []
        if batch:
            batch_comments = self.batch_process_comments(batch)
            comments.extend(batch_comments)
        
        return comments
    
    def batch_process_comments(self, comments: List[Dict[str, Any]]):
        """
        Batch processing of comments in contrast to one request per comment (bottleneck)
        """
        # How can I convey is_reply and head_comment_id?
        # parent_id
        processed_comments = []
        
        logger.info("Processing batch")
        
        # Get channel details by batch
        # Extract author channel IDs from authorChannelId field
        channel_ids = []
        for comment in comments:
            author_channel_id = comment["snippet"]["authorChannelId"]
            if isinstance(author_channel_id, dict):
                channel_ids.append(author_channel_id["value"])
            else:
                channel_ids.append(author_channel_id)

        account_details_response = self.get_commenter_details(channel_ids)
        if account_details_response["status_code"] == 200:
            account_details = account_details_response["data"]["items"]
            logger.info(f"Received {len(account_details)} results")
        else:
            raise Exception("Unable to process batch")

        # Create a mapping of channel ID to account details for efficient lookup
        account_details_map = {detail["id"]: detail for detail in account_details}

        for comment in comments:
            # video_channel_id should come from the video snippet we added
            video_channel_id = comment.get("videoChannelId", comment["snippet"].get("videoOwnerChannelId", ""))

            # Extract author channel ID
            author_channel_id_raw = comment["snippet"]["authorChannelId"]
            author_channel_id = author_channel_id_raw["value"] if isinstance(author_channel_id_raw, dict) else author_channel_id_raw

            author_display_name = comment["snippet"]["authorDisplayName"]

            # Get account details from the map
            account_detail = account_details_map.get(author_channel_id)
            if not account_detail:
                logger.warning(f"No account details found for {author_display_name} ({author_channel_id})")
                continue
            video_id = comment["videoId"]
            video_title = comment["videoTitle"]
            video_publish_date = comment["videoPublishDate"]
            
            comment_id = comment["id"]
            head_comment_id = comment.get("parentId", None)
            is_reply = head_comment_id != "" or head_comment_id != None
            published_at = comment["snippet"]["publishedAt"]
            updated_at = comment["snippet"]["updatedAt"]
            like_count = comment["snippet"]["likeCount"]
            text = comment["snippet"]["textOriginal"]
            
            author_created_at = account_detail["snippet"]["publishedAt"]
            is_hidden_sub_count = account_detail["statistics"]["hiddenSubscriberCount"]
            author_sub_count = account_detail["statistics"]["subscriberCount"] \
                if not is_hidden_sub_count else -1
            author_video_count = account_detail["statistics"]["videoCount"]
            
            comment_data = CommentData(
                video_channel_id = video_channel_id,
                
                author_channel_id = author_channel_id,
                author_display_name = author_display_name,
                author_created_at = author_created_at,
                author_sub_count = author_sub_count,
                author_video_count = author_video_count,
                
                video_id = video_id,
                video_title = video_title,
                video_publish_date = video_publish_date,
                
                comment_id = comment_id,
                is_reply = is_reply,
                thread_id = head_comment_id,
                published_at = published_at,
                updated_at = updated_at,
                like_count = like_count,
                text = text,
            )
            
            processed_comments.append(comment_data)
        
        logger.info("Finished processing batch")
            
        return processed_comments

    def process_comment(self, comment: Dict[str, Any], 
                        video_id: str, video_title: str, 
                        video_publish_date: str | datetime,
                        is_reply: bool, 
                        head_comment_id: Optional[str] = None) -> CommentData:
        """
        Parses a single comment from a comment thread
        
        Args:
            comment: The Comment Resource (https://developers.google.com/youtube/v3/docs/comments#resource)
            video_id: The underlying video in which the comment exists
            is_reply: Whether or not the comment is a reply within a thread
            
        Returns:
            CommentData: The aggregated data retrieved
        """
        snippet = comment["snippet"]
        comment_id = comment["id"]
        
        author_display_name = snippet["authorDisplayName"]
        like_count = snippet["likeCount"]
        text = snippet["textOriginal"]
        author_channel_id = snippet["authorChannelId"]["value"]
        video_channel_id = snippet["channelId"]
        updated_at = snippet["updatedAt"]
        published_at = snippet["publishedAt"]
        
        account_details_response = self.get_commenter_details(account_id=author_channel_id)
        
        if account_details_response["status_code"] == 200:
            account_details = account_details_response["data"]["items"][0]
            
            author_created_at = account_details["snippet"]["publishedAt"]
            is_hidden_sub_count = account_details["statistics"]["hiddenSubscriberCount"]
            author_sub_count = account_details["statistics"]["subscriberCount"] \
                if not is_hidden_sub_count else -1
            author_video_count = account_details["statistics"]["videoCount"]
            
            return CommentData(
                video_channel_id = video_channel_id,
                
                author_channel_id = author_channel_id,
                author_display_name = author_display_name,
                author_created_at = author_created_at,
                author_sub_count = author_sub_count,
                author_video_count = author_video_count,
                
                video_id = video_id,
                video_title = video_title,
                video_publish_date = video_publish_date,
                
                comment_id = comment_id,
                is_reply = is_reply,
                thread_id = head_comment_id,
                published_at = published_at,
                updated_at = updated_at,
                like_count = like_count,
                text = text,
            )
        else:
            logger.warning(f"Unable to retrieve commenter account info for: {author_display_name}")
            return CommentData(
                author_display_name = author_display_name,
                author_channel_id = author_channel_id,
                like_count = like_count,
                text = text,
                video_id = video_id,
                video_channel_id = video_channel_id,
                updated_at = updated_at,
                published_at = published_at,
                is_reply = is_reply,
                head_comment_id = head_comment_id,
            )