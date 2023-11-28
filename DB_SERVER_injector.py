
# 커넥터 부르고 
from DB_SERVER_connector import connector
conn, cursor = connector()

def injector(datas):
    
    # 데이터 삽입 쿼리
    query = "INSERT INTO your_table (column1, column2) VALUES (%s, %s)"

    # 쿼리 실행
    cursor.execute(query, datas)

    # 변경사항 저장
    conn.commit()

    # 연결 종료
    cursor.close()
    conn.close()

# pg_hba.conf 파일에서 외부에서 접근 가능하도록 설정해줘야함