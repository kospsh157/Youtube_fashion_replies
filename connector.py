import psycopg2

# 데이터베이스 설정
DATABASE = {
    'dbname': 'your_dbname',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',  # or your host
    'port': '5432'       # or your port
}

# 데이터베이스 연결
connection = psycopg2.connect(**DATABASE)
cursor = connection.cursor()

connection.commit()

# 데이터 삽입
name = "John Doe"
age = 30
cursor.execute("INSERT INTO my_table (name, age) VALUES (%s, %s)", (name, age))
connection.commit()

# 연결 종료
cursor.close()
connection.close()


'''
    비디오_id,
    제목,
    채널명,
    채널_아이디,
    댓글,
    시청수,
    좋아요,
    싫어요

'''
