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

# 테이블 생성 (예제용)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS my_table (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        age INTEGER
    )
""")
connection.commit()

# 데이터 삽입
name = "John Doe"
age = 30
cursor.execute("INSERT INTO my_table (name, age) VALUES (%s, %s)", (name, age))
connection.commit()

# 연결 종료
cursor.close()
connection.close()
