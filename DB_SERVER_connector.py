# 로컬 외부 서버용
import psycopg2


def connentor():
    # 데이터베이스 연결 설정
    db_config = {
        'host': '192.168.1.101',  # 데이터베이스 서버의 IP 주소
        'dbname': 'youtube_videos',
        'user': 'psh0826',
        'password': '15243',
        'port': 5432  # PostgreSQL의 기본 포트는 5432입니다
    }

    # 데이터베이스에 연결
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    return (conn, cursor)
