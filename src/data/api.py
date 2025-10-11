"""
File: src/data/api.py

This file details modules designed to scrape data from the Youtube API
"""

from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

from dotenv import load_dotenv
import os
from pathlib import Path

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from datetime import datetime

from functools import wraps

from pprint import pprint

class CommentData(TypedDict):
    comment_id: str
    author_display_name: str
    author_channel_id: str
    author_created_date: str | datetime
    video_channel_id: str
    video_id: str
    video_title: str
    video_publish_date: str | datetime
    channel_id: str
    text: str
    like_count: int
    updated_at: str | datetime
    published_at: str | datetime
    is_reply: bool
    author_created_at: str | datetime
    author_sub_count: int
    author_video_count: int

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

# if any([var is None for var in [API_KEY, API_VERSION, API_SERVICE_NAME]]):
#     raise EnvironmentError(f"Failed to load environment variables; \
#         Ensure the environment variables match API_KEY, API_VERSION, API_SERVICE_NAME")

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
            print(f"HTTP Error {status_code}: {error_content}")
            
            if status_code == 403:
                print("Comments disabled or quota exceeded")
            elif status_code == 404:
                print("Video not found")
            elif status_code == 400:
                print("Invalid video ID")
            
            return {
                "status_code": status_code
            }
        
    return wrapper

@api_status_handler
def get_channel(client: Resource, handle: str):
    """
    Retrieves the Channel Resource given the handle
    (https://developers.google.com/youtube/v3/docs/channels/list)
    """
    # Create request via the Google API (This is a quite convenient abstraction)
    request = client.channels().list(
        part="snippet,contentDetails,statistics",
        forHandle=handle
    )

    channel_response = request.execute()
    return channel_response

@api_status_handler
def get_channel_uploads(client: Resource, uploads_playlist_id: str):
    """
    Retrieves the uploads playlist given the uploads ID
    (https://developers.google.com/youtube/v3/docs/playlistItems/list)
    """
    request = client.playlistItems().list(
        part="contentDetails,id,snippet,status",
        playlistId=uploads_playlist_id,
    )

    uploads_response = request.execute()
    return uploads_response

@api_status_handler
def get_comment_thread(client: Resource, video_id: str):
    """
    Retrieves the comment thread given the video ID
    (https://developers.google.com/youtube/v3/docs/commentThreads/list)
    """
    # Get the comment thread
    request = client.commentThreads().list(
        part="id,replies,snippet",
        videoId=video_id
    )
    comment_thread_response = request.execute()
    return comment_thread_response

@api_status_handler
def get_commenter_details(client: Resource, account_id: str):
    request = client.channels().list(
        part="id,snippet,statistics",
        id=account_id
    )
    response = request.execute()
    
    return response

def scraper() -> List[CommentData]:
    """
    Retrieves comments from the 5 most recent videos from all channels.
    
    Returns:
        List[CommentData]: A list of CommentData TypedDicts
    """
    all_data = []
    for channel_handle in CHANNEL_HANDLES:
        print(f"="*80)
        print(f"\tRetrieving comments from: {channel_handle:^30}")
        print(f"="*80)
        
        # Get channel info
        channel_response = get_channel(channel_handle)
        if channel_response["status_code"] == 200:
            channel = channel_response["data"]
        else:
            print(f"Unable to retrieve channel data; skipping...")
            continue
        
        # Retrieve uploads ID and get videos
        playlist_id = channel["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        uploads_response = get_channel_uploads(playlist_id)     # Should retrieve 5 videos per page, 
                                                                #   as per default pagination settings                      
        if uploads_response["status_code"] == 200:
            uploads = uploads_response["data"]["items"]
        else:
            print(f"Unable to process {channel_handle} uploads playlist; skipping...")
            continue
        
        comments = process_videos(uploads)
        all_data.extend(comments)
        
    return all_data
        

def process_videos(uploads: List[Dict[str, Any]]) -> List[CommentData]:
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
        
        # Use video ID for comment thread
        comment_thread_response = get_comment_thread(video_id)
        if comment_thread_response["status_code"] == 200:
            comment_thread = comment_thread_response["data"]["items"]
        else:
            print(f"Video [{video_id}] has comments disabled. Skipping...")
            continue
        
        # Retrieve all comments from video's comment thread
        video_comments = process_comment_thread(comment_thread, 
                                                video_id, 
                                                video_title,
                                                video_publish_date)
        comments.extend(video_comments)
        
    return comments


def process_comment_thread(comment_thread: List[Dict[str, Any]],
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
    for head_comment in comment_thread:
        # Get video ID
        video_id = head_comment["snippet"]["videoId"]
        
        # Get head comment's comment data
        comment = head_comment["snippet"]["topLevelComment"]
        comment = process_comment(comment, video_id=video_id, is_reply=False)
        comments.append(comment)
        
        # Get head comment's ID (for replies)
        head_comment_id = head_comment["snippet"]["topLevelComment"]["id"]
        
        # If the head comment has replies, get those comments
        replies = head_comment.get("replies", {})
        thread_replies = replies.get("comments", [])
        for reply in thread_replies:
            comment = process_comment(reply, video_id, 
                                      video_title, 
                                      video_publish_date, 
                                      is_reply=True, head_comment_id=head_comment_id)
            comments.append(comment)
    
    return comments

def process_comment(comment: Dict[str, Any], 
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
    is_updated = updated_at == published_at
    
    account_details_response = get_commenter_details(account_id=author_channel_id)
    
    if account_details_response["status_code"] == 200:
        account_details = account_details_response["data"]["items"][0]
        
        author_created_at = account_details["snippet"]["publishedAt"],
        is_hidden_sub_count = account_details["statistics"]["hiddenSubscriberCount"]
        author_sub_count = account_details["statistics"]["subscriberCount"] \
            if not is_hidden_sub_count else 0
        author_video_count = account_details["statistics"]["videoCount"]
        
        return CommentData(
            comment_id = comment_id,
            author_display_name = author_display_name,
            author_channel_id = author_channel_id,
            like_count = like_count,
            text = text,
            video_id = video_id,
            video_title = video_title,
            video_publish_date = video_publish_date,
            video_channel_id = video_channel_id,
            updated_at = updated_at,
            published_at = published_at,
            is_updated = is_updated,
            is_reply = is_reply,
            head_comment_id = head_comment_id,
            author_created_at = author_created_at,
            author_is_hidden_sub_count = is_hidden_sub_count,
            author_sub_count = author_sub_count,
            author_video_count = author_video_count
        )
    else:
        print(f"Unable to retrieve commenter account info for: {author_display_name}")
        return CommentData(
            author_display_name = author_display_name,
            author_channel_id = author_channel_id,
            like_count = like_count,
            text = text,
            video_id = video_id,
            video_channel_id = video_channel_id,
            updated_at = updated_at,
            published_at = published_at,
            is_updated = is_updated,
            is_reply = is_reply,
            head_comment_id = head_comment_id,
        )