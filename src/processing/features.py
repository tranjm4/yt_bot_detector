"""
File: src/processing/features.py

This file contains modules used for feature engineering
"""

import pandas as pd
import numpy as np

from typing import Optional, List


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

def get_comment_hour_stats(psql_client, min_comments: Optional[int] = 2) -> pd.DataFrame:
    """
    Computes circular mean hour and hour deviance statistics for user comments.
    Uses circular statistics to properly handle the 24-hour wraparound.

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of comments required per user

    Returns:
        pd.DataFrame: With columns ["userId", "circular_mean_hour", "mean_hour_deviance",
                                     "std_hour_deviance", "comment_count"]
    """
    assert min_comments is None or isinstance(min_comments, int)

    query = """
    WITH user_hours AS (
        SELECT
            c.commenterId,
            EXTRACT(HOUR FROM c.publishDate) AS comment_hour
        FROM Yt.Comments AS c
    ),
    user_counts AS (
        SELECT
            commenterId,
            COUNT(*) AS comment_count
        FROM user_hours
        GROUP BY commenterId
        HAVING COUNT(*) >= %s
    )
    SELECT
        uh.commenterId,
        uh.comment_hour
    FROM user_hours uh
    INNER JOIN user_counts uc
        ON uh.commenterId = uc.commenterId
    ORDER BY uh.commenterId;
    """

    min_comments = min_comments or 2
    result = psql_client.query(query, (min_comments,))

    if not result:
        return pd.DataFrame(columns=["userId", "circular_mean_hour", "mean_hour_deviance",
                                    "std_hour_deviance", "commentCount"])

    df = pd.DataFrame(result, columns=["userId", "comment_hour"])

    # Convert comment_hour to float (PostgreSQL returns Decimal)
    df["comment_hour"] = df["comment_hour"].astype(float)

    # Compute per-user statistics
    user_stats = []
    for user_id, group in df.groupby("userId"):
        hours = group["comment_hour"]

        # Compute circular mean
        circ_mean = compute_circular_mean_hours(hours)

        # Compute hour deviances
        deviances = compute_hour_deviance(hours, circ_mean)

        user_stats.append({
            "userId": user_id,
            "circular_mean_hour": circ_mean,
            "mean_hour_deviance": deviances.mean(),
            "std_hour_deviance": deviances.std(),
            "comment_count": len(hours)
        })

    return pd.DataFrame(user_stats)

def get_reply_ratio(psql_client, min_comments: Optional[int] = None) -> pd.DataFrame:
    """
    Computes reply ratio statistics for users.

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of comments required per user

    Returns:
        pd.DataFrame: With columns ["userId", "reply_count", "toplevel_count", "reply_ratio"]
    """
    assert min_comments is None or isinstance(min_comments, int)

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
    {having_clause}
    ORDER BY reply_ratio DESC NULLS LAST;
    """

    if min_comments is not None:
        having_clause = f"HAVING COUNT(*) >= {int(min_comments)}"
    else:
        having_clause = ""

    query = query.format(having_clause=having_clause)

    return pd.DataFrame(psql_client.query(query),
                       columns=["userId", "reply_count", "toplevel_count", "reply_ratio"])

def get_comment_count(psql_client, min_comments: Optional[int] = None) -> pd.DataFrame:
    """
    Computes the comment count
    
    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): A minimum threshold amount of comments, if applicable
        
    Returns:
        pd.DataFrame: With columns "userId" and "comment_count"
    """
    # Protect from SQL injection
    assert min_comments is None or isinstance(min_comments, int)

    query = f"""
    SELECT c.commenterId, COUNT(c.commenterId) AS comment_count
    FROM Yt.Comments AS c
    GROUP BY c.commenterId
    ORDER BY comment_count DESC
    {'LIMIT %s' if min_comments else ''};
    """
    if isinstance(min_comments, int):
        return pd.DataFrame(psql_client.query(query, (min_comments,)),
                            columns=["userId", "comment_count"])
    else:
        return pd.DataFrame(psql_client.query(query),
                            columns=["userId", "comment_count"])
        
def get_creation_dates(psql_client) -> pd.DataFrame:
    """
    Computes the account creation date
    """

    query = f"""
    SELECT c.commenterId, u.createDate
    FROM Yt.Comments AS c
    JOIN Yt.Users AS u
        ON c.commenterId = u.userId
    ORDER BY u.createDate DESC;
    """

    return pd.DataFrame(psql_client.query(query),
                        columns=["userId", "account_creation_date"])

def get_comment_latency(psql_client, min_comments: Optional[int] = 2,
                       stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes statistics on comment latency (time from video publish to comment).
    Filters out negative latencies (data quality issues) and users with < min_comments.

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of valid comments required per user
        stats (Optional[List[str]]): Which statistics to compute. Options:
                                     'mean', 'std', 'median', 'min', 'max', 'counts'
                                     Default: ['mean', 'std', 'counts']

    Returns:
        pd.DataFrame: With userId and selected statistic columns
    """
    assert min_comments is None or isinstance(min_comments, int)

    # Default stats
    if stats is None:
        stats = ['mean', 'std', 'counts']

    # Build SQL SELECT clause based on requested stats
    select_parts = ["c.commenterId"]
    columns = ["userId"]

    if 'mean' in stats:
        select_parts.append("AVG(EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) / 60) AS mean_latency_minutes")
        columns.append("mean_latency_minutes")

    if 'std' in stats:
        select_parts.append("STDDEV(EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) / 60) AS std_latency_minutes")
        columns.append("std_latency_minutes")

    if 'median' in stats:
        select_parts.append("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) / 60) AS median_latency_minutes")
        columns.append("median_latency_minutes")

    if 'min' in stats:
        select_parts.append("MIN(EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) / 60) AS min_latency_minutes")
        columns.append("min_latency_minutes")

    if 'max' in stats:
        select_parts.append("MAX(EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) / 60) AS max_latency_minutes")
        columns.append("max_latency_minutes")

    if 'counts' in stats:
        select_parts.append("COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) >= 0) AS valid_comment_count")
        select_parts.append("COUNT(*) FILTER (WHERE EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) < 0) AS filtered_comment_count")
        columns.extend(["valid_comment_count", "filtered_comment_count"])

    query = f"""
    SELECT
        {', '.join(select_parts)}
    FROM Yt.Comments AS c
    JOIN Yt.Videos AS v
        ON c.videoId = v.videoId
    WHERE EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) >= 0
    GROUP BY c.commenterId
    HAVING COUNT(*) >= %s
    ORDER BY c.commenterId;
    """

    min_comments = min_comments or 2
    result = psql_client.query(query, (min_comments,))

    return pd.DataFrame(result, columns=columns)
    
def get_comment_latency_variance_score(psql_client, min_comments: Optional[int] = 2) -> pd.DataFrame:
    """
    Computes a weighted variance score for latency that emphasizes low variance with high sample size.

    Score = log(comment_count) / (std_latency + epsilon)

    Higher score = more suspicious (low variance with many comments)

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of valid comments required per user

    Returns:
        pd.DataFrame: With columns ["userId", "latency_variance_score"]
    """
    assert min_comments is None or isinstance(min_comments, int)

    epsilon = 0.1  # Avoid division by zero

    query = f"""
    SELECT
        c.commenterId,
        LN(COUNT(*)) / (STDDEV(EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) / 60) + {epsilon}) AS latency_variance_score
    FROM Yt.Comments AS c
    JOIN Yt.Videos AS v
        ON c.videoId = v.videoId
    WHERE EXTRACT(EPOCH FROM (c.publishDate - v.publishDate)) >= 0
    GROUP BY c.commenterId
    HAVING COUNT(*) >= %s
    ORDER BY latency_variance_score DESC;
    """

    min_comments = min_comments or 2
    result = psql_client.query(query, (min_comments,))

    return pd.DataFrame(result, columns=["userId", "latency_variance_score"])
    

def get_channel_diversity(psql_client, min_comments: Optional[int] = 2) -> pd.DataFrame:
    """
    Computes channel diversity: number of unique channels a user comments on.

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of comments required

    Returns:
        pd.DataFrame: With columns ["userId", "unique_channel_count", "comment_count"]
    """
    assert min_comments is None or isinstance(min_comments, int)

    query = """
    SELECT
        c.commenterId,
        COUNT(DISTINCT v.channelId) AS unique_channel_count,
        COUNT(*) AS comment_count
    FROM Yt.Comments AS c
    JOIN Yt.Videos AS v
        ON c.videoId = v.videoId
    GROUP BY c.commenterId
    HAVING COUNT(*) >= %s
    ORDER BY unique_channel_count DESC;
    """

    min_comments = min_comments or 2
    result = psql_client.query(query, (min_comments,))

    return pd.DataFrame(result,
                       columns=["userId", "unique_channel_count", "comment_count"])

def get_text_length_stats(psql_client, min_comments: Optional[int] = 2) -> pd.DataFrame:
    """
    Computes mean and std dev of comment text length per user.

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of comments required

    Returns:
        pd.DataFrame: With columns ["userId", "mean_text_length", "std_text_length", "comment_count"]
    """
    assert min_comments is None or isinstance(min_comments, int)

    query = """
    SELECT
        c.commenterId,
        AVG(LENGTH(c.commentText)) AS mean_text_length,
        STDDEV(LENGTH(c.commentText)) AS std_text_length,
        COUNT(*) AS comment_count
    FROM Yt.Comments AS c
    GROUP BY c.commenterId
    HAVING COUNT(*) >= %s
    ORDER BY mean_text_length DESC;
    """

    min_comments = min_comments or 2
    result = psql_client.query(query, (min_comments,))

    return pd.DataFrame(result,
                       columns=["userId", "mean_text_length", "std_text_length", "comment_count"])

def get_text_repetition_stats(psql_client, min_comments: Optional[int] = 2) -> pd.DataFrame:
    """
    Computes text repetition statistics: how often users copy-paste the same comments.

    Metrics:
    - unique_comment_ratio: ratio of unique comments to total comments
    - max_duplicate_count: highest number of times any single comment was repeated
    - avg_duplicate_count: average repetitions for duplicated comments only
    - total_duplicates: total number of duplicate comment instances

    Args:
        psql_client (psql.Psql): The client connection to the Postgres database
        min_comments (Optional[int]): Minimum number of comments required

    Returns:
        pd.DataFrame: With columns ["userId", "unique_comment_ratio", "max_duplicate_count",
                                     "avg_duplicate_count", "total_duplicates", "comment_count"]
    """
    assert min_comments is None or isinstance(min_comments, int)

    query = """
    WITH comment_counts AS (
        SELECT
            commenterId,
            commentText,
            COUNT(*) as duplicate_count
        FROM Yt.Comments
        GROUP BY commenterId, commentText
    ),
    user_stats AS (
        SELECT
            commenterId,
            COUNT(*) AS totalComments,
            COUNT(DISTINCT commentText) AS uniqueComments,
            MAX(duplicate_count) AS maxDuplicates,
            AVG(CASE WHEN duplicate_count > 1 THEN duplicate_count ELSE NULL END) AS avgDuplicates,
            SUM(CASE WHEN duplicate_count > 1 THEN duplicate_count ELSE 0 END) AS totalDuplicates
        FROM comment_counts
        GROUP BY commenterId
    )
    SELECT
        commenterId,
        CAST(uniqueComments AS FLOAT) / NULLIF(totalComments, 0) AS unique_comment_ratio,
        maxDuplicates AS max_duplicate_count,
        avgDuplicates AS avg_duplicate_count,
        totalDuplicates AS total_duplicates,
        totalComments AS comment_count
    FROM user_stats
    WHERE totalComments >= %s
    ORDER BY unique_comment_ratio ASC;
    """

    min_comments = min_comments or 2
    result = psql_client.query(query, (min_comments,))

    return pd.DataFrame(result,
                       columns=["userId", "unique_comment_ratio", "max_duplicate_count",
                               "avg_duplicate_count", "total_duplicates", "comment_count"])

# ============================================================================
# Graph-based features
# ============================================================================
# Note: These features require networkx and can be computationally expensive.
# Consider running them separately or on a subset of data.

def build_user_video_graph(psql_client, sample_size: Optional[int] = None):
    """
    Builds a bipartite graph of users and videos they comment on.

    Args:
        psql_client: Database connection
        sample_size (Optional[int]): If provided, randomly sample this many user-video pairs

    Returns:
        Tuple of (networkx.Graph, list of user nodes, list of video nodes)

    Raises:
        ValueError: If no data is available or graph would be invalid
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for graph features. Install with: pip install networkx")

    if sample_size:
        query = """
        SELECT commenterId, videoId
        FROM (
            SELECT DISTINCT c.commenterId, c.videoId
            FROM Yt.Comments AS c
        ) AS distinct_pairs
        ORDER BY RANDOM()
        LIMIT %s;
        """
        pairs = psql_client.query(query, (int(sample_size),))
    else:
        query = """
        SELECT DISTINCT c.commenterId, c.videoId
        FROM Yt.Comments AS c;
        """
        pairs = psql_client.query(query)

    if not pairs:
        raise ValueError("No user-video pairs found in database")

    users = list(set(pair[0] for pair in pairs))
    videos = list(set(pair[1] for pair in pairs))

    if not users or not videos:
        raise ValueError(f"Invalid graph structure: {len(users)} users, {len(videos)} videos")

    G = nx.Graph()
    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(videos, bipartite=1)
    G.add_edges_from(pairs)

    return G, users, videos

def get_community_assignments(psql_client, sample_size: Optional[int] = None,
                              resolution: float = 1.0) -> pd.DataFrame:
    """
    Detects communities using the Louvain method on the user projection graph.
    Users who comment on similar videos are likely to be in the same community.

    WARNING: This is computationally expensive. Use sample_size to limit data.

    Args:
        psql_client: Database connection
        sample_size (Optional[int]): Number of user-video pairs to sample
        resolution (float): Resolution parameter for Louvain algorithm (higher = more communities)

    Returns:
        pd.DataFrame: With columns ["userId", "communityId", "communitySize"]
    """
    try:
        import networkx as nx
        from networkx.algorithms import bipartite
    except ImportError:
        raise ImportError("networkx is required for graph features. Install with: pip install networkx")

    # Build bipartite graph
    G, users, _ = build_user_video_graph(psql_client, sample_size)

    # Project to user graph (users connected if they comment on same videos)
    user_graph = bipartite.weighted_projected_graph(G, users)

    # Detect communities
    communities = nx.community.louvain_communities(user_graph, resolution=resolution)

    # Create mapping of user to community
    user_community_map = {}
    community_sizes = {}

    for comm_id, community in enumerate(communities):
        community_sizes[comm_id] = len(community)
        for user in community:
            user_community_map[user] = comm_id

    # Convert to DataFrame
    data = [
        {
            "userId": user,
            "communityId": comm_id,
            "communitySize": community_sizes[comm_id]
        }
        for user, comm_id in user_community_map.items()
    ]

    return pd.DataFrame(data)

def get_pagerank_scores(psql_client, sample_size: Optional[int] = None,
                       alpha: float = 0.85) -> pd.DataFrame:
    """
    Computes PageRank scores for users based on the user projection graph.
    Higher PageRank indicates more "influential" users in the comment network.

    WARNING: This is computationally expensive. Use sample_size to limit data.

    Args:
        psql_client: Database connection
        sample_size (Optional[int]): Number of user-video pairs to sample
        alpha (float): Damping parameter for PageRank (default 0.85)

    Returns:
        pd.DataFrame: With columns ["userId", "pagerank"]
    """
    try:
        import networkx as nx
        from networkx.algorithms import bipartite
    except ImportError:
        raise ImportError("networkx is required for graph features. Install with: pip install networkx")

    # Build bipartite graph
    G, users, _ = build_user_video_graph(psql_client, sample_size)

    # Project to user graph
    user_graph = bipartite.weighted_projected_graph(G, users)

    # Compute PageRank
    pagerank_scores = nx.pagerank(user_graph, alpha=alpha)

    # Convert to DataFrame
    data = [
        {"userId": user, "pagerank": score}
        for user, score in pagerank_scores.items()
    ]

    return pd.DataFrame(data).sort_values("pagerank", ascending=False)

def get_temporal_co_commenting_clusters(psql_client, time_window_minutes: int = 60,
                                        min_co_occurrences: int = 3) -> pd.DataFrame:
    """
    Identifies users who repeatedly comment within similar timeframes across multiple videos.
    This can detect coordinated behavior or bot networks.

    Args:
        psql_client: Database connection
        time_window_minutes (int): Time window to consider comments as "co-occurring"
        min_co_occurrences (int): Minimum number of videos where users co-comment

    Returns:
        pd.DataFrame: With columns ["userId1", "userId2", "coCommentCount", "sharedVideos"]
    """
    query = f"""
    WITH user_comment_times AS (
        SELECT
            c1.commenterId AS userId1,
            c2.commenterId AS userId2,
            c1.videoId,
            c1.publishDate AS time1,
            c2.publishDate AS time2
        FROM Yt.Comments c1
        JOIN Yt.Comments c2
            ON c1.videoId = c2.videoId
            AND c1.commenterId < c2.commenterId  -- Avoid duplicates and self-joins
            AND ABS(EXTRACT(EPOCH FROM (c1.publishDate - c2.publishDate)) / 60) <= %s
    )
    SELECT
        userId1,
        userId2,
        COUNT(DISTINCT videoId) AS coCommentCount,
        ARRAY_AGG(DISTINCT videoId) AS sharedVideos
    FROM user_comment_times
    GROUP BY userId1, userId2
    HAVING COUNT(DISTINCT videoId) >= %s
    ORDER BY coCommentCount DESC;
    """

    result = psql_client.query(query, (time_window_minutes, min_co_occurrences))

    return pd.DataFrame(result,
                       columns=["userId1", "userId2", "coCommentCount", "sharedVideos"])