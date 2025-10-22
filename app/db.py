"""
Database connection utilities for Streamlit app.

Handles connection pooling and query execution for PostgreSQL.
"""

import os
from typing import List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

from dotenv import load_dotenv
import os

# Load .env.railway from project root
_current_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_current_dir, "..", ".env.railway")
load_dotenv(_env_path)


class DatabaseConnection:
    """Manages PostgreSQL database connections with connection pooling."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 5,
    ):
        """
        Initialize database connection pool.

        Args:
            host: Database host (defaults to POSTGRES_HOST env var)
            port: Database port (defaults to POSTGRES_PORT env var)
            user: Database user (defaults to POSTGRES_USER env var)
            password: Database password (defaults to POSTGRES_PASSWORD env var)
            database: Database name (defaults to POSTGRES_DB env var)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.user = user or os.getenv("POSTGRES_USER")
        self.password = password or os.getenv("POSTGRES_PASSWORD")
        self.database = database or os.getenv("POSTGRES_DB")

        if not all([self.user, self.password, self.database]):
            raise ValueError(
                "Database credentials not provided. Set POSTGRES_USER, "
                "POSTGRES_PASSWORD, and POSTGRES_DB environment variables."
            )

        # Create connection pool
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            min_connections,
            max_connections,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.

        Usage:
            with db.get_connection() as conn:
                # Use connection
        """
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as a DataFrame.

        Args:
            query: SQL query to execute
            params: Query parameters (optional)

        Returns:
            DataFrame with query results
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query to execute
            params: Query parameters (optional)

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rowcount = cursor.rowcount
            conn.commit()
            cursor.close()
            return rowcount

    def close_all(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()


# Queries for predictions

def get_latest_predictions(
    db: DatabaseConnection, limit: int = 100
) -> pd.DataFrame:
    """
    Get the latest prediction run results.

    Args:
        db: Database connection
        limit: Maximum number of predictions to return

    Returns:
        DataFrame with prediction results
    """
    query = """
        SELECT
            p.predictionId,
            p.userId,
            u.username,
            u.createDate as accountCreateDate,
            u.subCount,
            u.videoCount,
            p.modelName,
            p.modelVersion,
            p.anomalyScore,
            p.label,
            p.runTimestamp,
            p.featureValues
        FROM YT.Predictions p
        JOIN YT.Users u ON p.userId = u.userId
        WHERE p.runTimestamp = (
            SELECT MAX(runTimestamp) FROM YT.Predictions
        )
        AND p.label = -1
        ORDER BY p.anomalyScore ASC
        LIMIT %s
    """
    return db.execute_query(query, (limit,))


def get_prediction_runs(db: DatabaseConnection, limit: int = 10) -> pd.DataFrame:
    """
    Get summary of recent prediction runs.

    Args:
        db: Database connection
        limit: Number of runs to return

    Returns:
        DataFrame with run summaries
    """
    query = """
        SELECT
            runTimestamp,
            modelName,
            modelVersion,
            COUNT(*) as anomalyCount,
            AVG(anomalyScore) as avgAnomalyScore,
            MIN(anomalyScore) as minAnomalyScore,
            MAX(anomalyScore) as maxAnomalyScore
        FROM YT.Predictions
        WHERE label = -1
        GROUP BY runTimestamp, modelName, modelVersion
        ORDER BY runTimestamp DESC
        LIMIT %s
    """
    return db.execute_query(query, (limit,))


def get_user_prediction_history(
    db: DatabaseConnection, user_id: str
) -> pd.DataFrame:
    """
    Get prediction history for a specific user.

    Args:
        db: Database connection
        user_id: User ID to query

    Returns:
        DataFrame with user's prediction history
    """
    query = """
        SELECT
            predictionId,
            modelName,
            modelVersion,
            anomalyScore,
            label,
            runTimestamp,
            featureValues
        FROM YT.Predictions
        WHERE userId = %s
        ORDER BY runTimestamp DESC
    """
    return db.execute_query(query, (user_id,))


def get_user_comments(
    db: DatabaseConnection, user_id: str, limit: int = 50
) -> pd.DataFrame:
    """
    Get recent comments from a specific user.

    Args:
        db: Database connection
        user_id: User ID to query
        limit: Maximum number of comments to return

    Returns:
        DataFrame with user's comments
    """
    query = """
        SELECT
            c.commentId,
            c.commentText,
            c.publishDate,
            c.likeCount,
            c.isReply,
            v.title as videoTitle,
            v.videoId,
            ch.channelName
        FROM YT.Comments c
        JOIN YT.Videos v ON c.videoId = v.videoId
        JOIN YT.Channels ch ON v.channelId = ch.channelId
        WHERE c.commenterId = %s
        ORDER BY c.publishDate DESC
        LIMIT %s
    """
    return db.execute_query(query, (user_id, limit))


def get_anomaly_trends(
    db: DatabaseConnection, days: int = 30
) -> pd.DataFrame:
    """
    Get anomaly detection trends over time.

    Args:
        db: Database connection
        days: Number of days to look back

    Returns:
        DataFrame with daily anomaly counts
    """
    query = """
        SELECT
            DATE(runTimestamp) as date,
            COUNT(DISTINCT runTimestamp) as runs,
            COUNT(*) as totalAnomalies,
            AVG(anomalyScore) as avgScore
        FROM YT.Predictions
        WHERE label = -1
        AND runTimestamp >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(runTimestamp)
        ORDER BY date DESC
    """
    return db.execute_query(query, (days,))
