import psycopg2
from config import DB_PASS


def get_db_connection():
    connection = psycopg2.connect(
        dbname="youtube_videos",
        user="psh0826",
        password=DB_PASS,
        host="localhost",
        port="5432"
    )
    return connection
