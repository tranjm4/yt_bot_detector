"""
File: src/data/psql.py

This module contains functions used to interface with the PostgreSQL database
"""

import psycopg2
import os

from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field

class VersionFields(BaseModel):
    versionName: str
    createdAt: datetime
    versionDescription: Optional[str] = None

class ChannelFields(BaseModel):
    channelId: str = Field(max_length=30)

class UserFields(BaseModel):
    userId: str = Field(max_length=50)
    username: str = Field(max_length=40)
    createDate: datetime
    subCount: Optional[int] = None
    videoCount: Optional[int] = None
    versionName: str = Field(max_length=24)
    updatedAt: Optional[datetime] = None

class VideoFields(BaseModel):
    videoId: str = Field(max_length=20)
    title: str
    publishDate: datetime
    channelId: str = Field(max_length=30)
    versionName: str = Field(max_length=24)
    updatedAt: Optional[datetime] = None

class CommentFields(BaseModel):
    commentId: str = Field(max_length=50)
    commenterId: str = Field(max_length=50)
    videoId: str = Field(max_length=20)
    isReply: bool
    threadId: Optional[str] = Field(max_length=50)
    publishDate: datetime
    editDate: Optional[datetime] = None
    likeCount: Optional[int] = None
    commentText: str
    versionName: str = Field(max_length=24)
    updatedAt: Optional[datetime] = None

class Psql:
    def __init__(self):
        self.connection = self.connect_to_db()
    
    def connect_to_db(self):
        try:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            user = os.getenv("POSTGRES_USER", "")
            password = os.getenv("POSTGRES_PASSWORD", "")
            db_name = os.getenv("POSTGRES_DB", "")
            connection = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=db_name
            )
        
            return connection
        except:
            raise psycopg2.errors.ConnectionException("Failed to connect to PSQL database")
    
    def close_db(self):
        self.connection.close()
        
    def insert(self, table: str, data: BaseModel, on_conflict: str = "NOTHING"):
        """
        Inserts values into a table given fields

        Args:
            table: Table name to insert into
            data: Pydantic model with data to insert
            on_conflict: "NOTHING" to ignore duplicates, "UPDATE" to update on conflict
        """
        self._verify_table(table)

        tables = {
            "Versions": VersionFields,
            "Channels": ChannelFields,
            "Users": UserFields,
            "Videos": VideoFields,
            "Comments": CommentFields
        }

        # Validate date matches the model
        if not isinstance(data, tables[table]):
            raise TypeError(f"Data mnust be {tables[table].__name__} for table {table}")

        # Convert data to dict and insert
        data_dict = data.model_dump()
        # Remove updatedAt if None - let database default handle it
        if "updatedAt" in data_dict and data_dict["updatedAt"] is None:
            del data_dict["updatedAt"]
        columns = ", ".join(data_dict.keys())
        placeholders = ", ".join(["%s"] * len(data_dict))

        # Build ON CONFLICT clause
        conflict_clause = "ON CONFLICT DO NOTHING"
        if on_conflict.upper() == "UPDATE":
            # Get primary keys for each table
            primary_keys = {
                "Versions": ["versionName"],
                "Channels": ["channelId"],
                "Users": ["userId"],
                "Videos": ["videoId"],
                "Comments": ["commentId"]
            }
            pk_cols = ", ".join(primary_keys[table])
            update_cols = [col for col in data_dict.keys() if col not in primary_keys[table]]
            # Add updatedAt to update clause if it's not already there
            if "updatedAt" not in update_cols and table in ["Users", "Videos", "Comments"]:
                update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols]) + ", updatedAt = CURRENT_TIMESTAMP"
            else:
                update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
            conflict_clause = f"ON CONFLICT ({pk_cols}) DO UPDATE SET {update_clause}"

        query = f"INSERT INTO Yt.{table} ({columns}) VALUES ({placeholders}) {conflict_clause}"

        with self.connection.cursor() as cursor:
            cursor.execute(query, list(data_dict.values()))
            self.connection.commit()

        return True
    
    def peek(self, table: str) -> Tuple:
        """
        Peeks a single value from the specified table

        Args:
            table (str): The table from which to peek

        Returns:
            Tuple: A single row from the table
        """
        # Validate table name to prevent SQL injection
        self._verify_table(table)

        with self.connection.cursor() as cursor:
            query = f"SELECT * FROM YT.{table} LIMIT 1;"
            cursor.execute(query)
            return cursor.fetchone()
        
    def _verify_table(self, table: str):
        """
        Checks for valid table name
        Raises a ValueError if it doesn't exist
        """
        allowed_tables = ["Versions", "Channels", "Users", "Videos", "Comments"]
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}")
        
    def get_user_comments(self, username) -> Dict[str, Dict]:
        """
        Given a username, gets all comments from the user
        """
        pass