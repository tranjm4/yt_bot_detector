"""
File: src/data/psql.py

This module contains functions used to interface with the PostgreSQL database
"""

import psycopg2
import os

from typing import Tuple

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
        
    def insert(self, table, fields):
        pass
    
    def peek(self, table: str) -> Tuple:
        """
        Peeks a single value from the specified table

        Args:
            table (str): The table from which to peek

        Returns:
            Tuple: A single row from the table
        """
        # Validate table name to prevent SQL injection
        allowed_tables = ['Channels', 'Users', 'Videos', 'Comments']
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}")

        with self.connection.cursor() as cursor:
            query = f"SELECT * FROM Yt.{table} LIMIT 1;"
            cursor.execute(query)
            return cursor.fetchone()