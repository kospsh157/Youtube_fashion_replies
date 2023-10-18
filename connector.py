import psycopg2

# 데이터베이스 설정
DATABASE = {
    'dbname': 'test_1',
    'user': 'psh0826',
    'password': '15243',
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
    title 
    published_at 
    views = item['statistics'].get('viewCount', 0)
    likes = item['statistics'].get('likeCount', 0)
    dislikes = item['statistics'].get('dislikeCount', 0)
    comments = item['statistics'].get('commentCount', 0)
    replys = item['replys']
'''
