"""
File: src/processing/features.py

This file contains modules used for feature engineering
"""

import pandas as pd
import numpy as np

from typing import Optional


def compute_circular_mean_hours(hours: pd.Series) -> np.float32:
    """
    Given a pandas series of hours, computes the circular mean,
    which accounts for the cyclical/wrap-around nature of time measurements.
    """
    radians = hours * (2 * np.pi / 24)
    sin_hour = np.sin(radians)
    cos_hour = np.cos(radians)
    
    sin_mean = np.mean(sin_hour)
    cos_mean = np.mean(cos_hour)
    
    circular_mean_radians = np.arctan2(sin_mean, cos_mean)
    circular_mean_hour = (circular_mean_radians * 24 / (2 * np.pi)) % 24
    return circular_mean_hour

def compute_hour_deviance(df: pd.Series, mean_hour: np.float32) -> pd.Series:
    return np.minimum((df - mean_hour) % 24, (mean_hour - df) % 24)

def get_reply_ratio(psql_client) -> pd.DataFrame:
    query = """
    SELECT c.commenterId,
        COUNT(*) FILTER (WHERE c.threadId IS NOT NULL) AS reply_count,
        COUNT(*) FILTER (WHERE c.threadId IS NULL) AS toplevel_count,
        CASE
            WHEN COUNT(*) FILTER (WHERE threadId IS NULL) = 0 THEN NULL
            ELSE COUNT(*) FILTER (WHERE threadId IS NOT NULL)::float /
                COUNT(*) FILTER (WHERE threadId IS NULL)
        END AS reply_ratio
    FROM Yt.Comments AS c
    GROUP BY c.commenterId
    ORDER BY reply_ratio DESC NULLS LAST
    LIMIT 10;
    """
    return pd.DataFrame(psql_client.query(query), columns=["userId", "replyRatio"])

def get_comment_count(psql_client, min_comments: Optional[int] = None) -> pd.DataFrame:
    """
    Computes the comment count
    
    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): A minimum threshold amount of comments, if applicable
        
    Returns:
        pd.DataFrame: With columns "userId" and "commentCount"
    """
    # Protect from SQL injection
    assert min_comments is None or isinstance(min_comments, int)
    
    query = f"""
    SELECT c.commenterId, COUNT(c.commenterId) AS commentCount
    FROM Yt.Comments AS c
    GROUP BY c.commenterId
    ORDER BY commentCount DESC
    {'LIMIT %s' if min_comments else ''};
    """
    if isinstance(min_comments, int):
        return pd.DataFrame(psql_client.query(query, (min_comments,)), 
                            columns=["userId", "commentCount"])
    else:
        return pd.DataFrame(psql_client.query(query),
                            columns=["userId", "commentCount"])
        
def get_creation_dates(psql_client) -> pd.DataFrame:
    """
    Computes the account creation date
    """
    
    query = f"""
    SELECT c.commenterId
    FROM Yt.Comments AS c
    JOIN Yt.Users AS u
        ON c.commenterId = u.userId
    ORDER BY u.createDate DESC;
    """
    
    return pd.DataFrame(psql_client.query(query),
                        columns=["userId", "accountCreationDate"])