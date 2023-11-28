# 외부 서버용
import psycopg2

def connentor():
    # 데이터베이스 연결 설정
    db_config = {
        'host': '192.168.1.xx',  # 데이터베이스 서버의 IP 주소
        'dbname': 'your_database',
        'user': 'your_username',
        'password': 'your_password',
        'port': 5432  # PostgreSQL의 기본 포트는 5432입니다
    }

    # 데이터베이스에 연결
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    return (conn, cursor)

