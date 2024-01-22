# 로컬 외부 서버용
import psycopg2
from config import DB_PASS


def DB_server_connentor():
    # 데이터베이스 연결 설정
    db_config = {
        'host': '192.168.0.101',  # 데이터베이스 서버의 IP 주소
        'dbname': 'youtube_videos',
        'user': 'psh0826',
        'password': DB_PASS,
        'port': 5432
    }

    # 데이터베이스에 연결
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    return (conn, cursor)


# 전체 과정
'''
    1. 일단 댓글을 긁어서 받아온다.
    2. 받아온 댓글을 전처리한다. 
    3. 전처리 과정이 끄나면 일단 로컬로 저장한다. 
    4. 로컬로 저장이 실패해도 그 다음에는 db서버에 접속해서 저장한다. 
    5. 로그 파일을 작성 하도록 한다. 나중에 로그 파일을 분석 할 수 있도록.


'''
