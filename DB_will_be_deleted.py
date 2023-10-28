from DB_connector import get_db_connection


def fetch_video_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    # video_datas 테이블에서 데이터를 가져오는 쿼리
    query = "SELECT video_id, title, time, views, likes, dislikes, comments_cnt, comments, query FROM video_datas"

    cursor.execute(query)

    results = cursor.fetchall()

    # 결과를 딕셔너리 형태로 변환
    videos = []
    for row in results:
        video_data = {
            'video_id': row[0],
            'title': row[1],
            'time': row[2],
            'views': row[3],
            'likes': row[4],
            'dislikes': row[5],
            'comments_cnt': row[6],
            'comments': row[7],
            'query': row[8]
        }
        videos.append(video_data)

    # 연결 종료
    cursor.close()
    conn.close()

    return videos


# 함수 호출
videos = fetch_video_data()
total_comments = 0
for video in videos:
    total_comments += len(video['comments'])
    print(video['comments'])
print(total_comments)
