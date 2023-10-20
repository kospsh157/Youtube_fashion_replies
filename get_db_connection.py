import psycopg2


def get_db_connection():
    connection = psycopg2.connect(
        dbname="youtube_videos",
        user="psh0826",
        password="15243",
        host="localhost",
        port="5432"
    )
    return connection
