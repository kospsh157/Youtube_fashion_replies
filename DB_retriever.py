from DB_connector import get_db_connection


def get_channel_ids():
    # 커서 생성
    connection = get_db_connection()
    cursor = connection.cursor()

    # 쿼리 실행
    cursor.execute("select id from channels")

    rows = cursor.fetchall()

    # 연결 종료
    cursor.close()
    connection.close()

    return rows


# 각 비디오당 가지고 있는 댓글수 및 댓글 추출
# 필요한 인자를 리스트에 담아서 보내야함
def select_query(columns):
    if not columns:
        raise ValueError("No columns provided for selection.")

    if isinstance(columns, (list, tuple)):
        columns = ", ".join(columns)
    elif isinstance(columns, str):
        pass
    else:
        raise TypeError(
            "Columns should be either a string or a list/tuple of strings.")

    conn = get_db_connection()
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(f'SELECT {columns} FROM video_datas')
            rows = cursor.fetchall()

    return rows


if __name__ == "__main__":
    channel_ids = get_channel_ids()
    print(channel_ids)
    print(len(channel_ids))

# db에서 튜플 형태로 각 항목을 가져오기 때문에
# 결과는 튜플이 담겨있는 리스트가 된다.
