"""
File: scripts/data/api.py

This file collects data from the Google Youtube API.
"""
from dotenv import load_dotenv
load_dotenv()

import psycopg2

from argparse import ArgumentParser
from datetime import datetime

from typing import List
from typing_extensions import TypedDict

from src.data.api import CommentData, YoutubeCommentScraper
from src.data.psql import Psql, VersionFields, ChannelFields, UserFields, VideoFields, CommentFields

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
    

def main(version_name, version_description):
    scraper = YoutubeCommentScraper()
    comment_data_list = scraper.scrape()
    logger.info(f"Collected comment_data_list: {comment_data_list}")
    process_comments(comment_data_list, version_name, version_description)
    
def process_comments(comments_list: List[CommentData], version_name: str, version_description: str):
    """
    Processes the list of retrieved comments to insert into the Postgres database
    
    Args:
        comments_list (List[CommentData]): The list of comments' data retrieved from src/data/api.scraper()
        version_name (str): The version name, provided by the commandline argument
        version_description (str): The version description, provided by the commandline argument
    """
    client = Psql()
    try:
        # Insert version into table
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        version_name = f"{version_name}:[{date}]"
        version_fields = _construct_version(version_name, date, version_description)
        
        client.insert("Versions", version_fields)
        
        for comment in comments_list:
            # Construct ChannelFields
            channel_fields = _construct_channel_fields(comment)
            result1 = client.insert("Channels", channel_fields)
            
            # Construct UserFields
            user_fields = _construct_user_fields(comment, version_name)
            result2 = client.insert("Users", user_fields)

            # Construct VideoFields
            video_fields = _construct_video_fields(comment, version_name)
            result3 = client.insert("Videos", video_fields)

            # Construct CommentFields
            comment_fields = _construct_comment_fields(comment, version_name)
            result4 = client.insert("Comments", comment_fields)
        
            if not all([result1, result2, result3, result4]):
                logger.error(f"Failed to insert into DB: "
                            f"\n\tChannels:\t{result1}"
                            f"\n\tUsers:\t{result2}"
                            f"\n\tVideos:\t{result3}"
                            f"\n\tComments:\t{result4}")
            else:
                logger.info("Successfully inserted comment data")
    except Exception as e:
        logger.error(f"Error processing comments: {e}")
        raise
    finally:
        client.close_db()
        
def _construct_version(version_name, date, description) -> VersionFields:
    """
    Constructs the VersionFields to be inserted into the PSQL Versions table
    
    Args:
        version_args: The commandline arguments (version, description)
        
    Returns:
        VersionFields: A BaseModel that complies with the PSQL Versions table
    """
    return VersionFields(
        versionName=version_name,
        createdAt=date,
        versionDescription=description
    )
    
def _construct_channel_fields(comment_data: CommentData) -> ChannelFields:
    """
    Constructs the ChannelFields to be inserted into the PSQL Channels table
    
    Args:
        comment_data (CommentData): The overall data retreived from the scraper
    
    Returns:
        ChannelFields: A BaseModel that complies with the PSQL Channels table
    """
    return ChannelFields(
        channelId=comment_data["video_channel_id"]
    )
    
def _construct_user_fields(comment_data: CommentData, version_name: str) -> UserFields:
    """
    Constructs the UserFields to be inserted into the PSQL Users table

    Args:
        comment_data (CommentData): The overall data retreived from the scraper
        version_name (str): The version name

    Returns:
        UserFields: A BaseModel that complies with the PSQL Users table
    """
    return UserFields(
        userId = comment_data["author_channel_id"],
        username = comment_data["author_display_name"],
        createDate = comment_data["author_created_at"],
        subCount = comment_data["author_sub_count"],
        videoCount = comment_data["author_video_count"],
        versionName = version_name
    )
    
def _construct_video_fields(comment_data: CommentData, version_name: str) -> VideoFields:
    """
    Constructs the VideoFields to be inserted into the PSQL Videos table
    
    Args:
        comment_data (CommentData): The overall data retreived from the scraper
        version_name (str): The version name
    
    Returns:
        VideoFields: A BaseModel that complies with the PSQL Videos table
    """
    return VideoFields(
        videoId = comment_data["video_id"],
        title = comment_data["video_title"],
        publishDate = comment_data["video_publish_date"],
        channelId = comment_data["video_channel_id"],
        versionName = version_name
    )
    
def _construct_comment_fields(comment_data: CommentData, version_name: str) -> CommentFields:
    """
    Constructs the CommentFields to be inserted into the PSQL Comments table
    
    Args:
        comment_data (CommentData): The overall data retreived from the scraper
        version_name (str): The version name
    
    Returns:
        CommentFields: A BaseModel that complies with the PSQL Comments table
    """
    return CommentFields(
        commentId = comment_data["comment_id"],
        commenterId = comment_data["author_channel_id"],
        videoId = comment_data["video_id"],
        isReply = comment_data["is_reply"],
        threadId = comment_data["thread_id"],
        publishDate = comment_data["published_at"],
        editDate = comment_data["updated_at"],
        likeCount = comment_data["like_count"],
        commentText = comment_data["text"],
        versionName = version_name
    )
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to be run for collecting comments")
    parser.add_argument("--version", required=True,
                        help="Version name for the script's runtime.")
    parser.add_argument("--description", required=True,
                        help="Description for the script's runtime.")
    args = parser.parse_args()
    main(args.version, args.description)