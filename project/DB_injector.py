from DB_connector import get_db_connection


'''
CREATE TABLE video_datas (
    video_id varchar(255) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    time timestamp NOT NULL,
    views BIGINT DEFAULT 0,
    likes BIGINT DEFAULT 0,
    dislikes BIGINT DEFAULT 0,
    comments_cnt BIGINT DEFAULT 0,
    comments TEXT DEFAULT '',
    query VARCHAR(255) NOT NULL
);

CREATE TABLE channels (
    name varchar(255) NOT NULL,
    id VARCHAR(255) PRIMARY KEY,
    subscribers int NOT NULL,
    topic varchar(255) NOT NULL
);

'''

# 유튜브에서 원시 데이터(댓글) 긁어온거 db에 저장하기 
# 여기서 파라미터query는 유튜브에서 영상을 '어떤 키워드로' 검색했는지다. 
# '패션'으로 검색을 했겠지만 필요에 의해선 다른 걸 검색해야 할수도 있으니..만들었던 것이다.
# connetor는 로컬과 로컬db서버를 구분하기위해 커넥터를 다르게 넣어줄수있도록 한것이다.
def insert_video_data(video_data_list, query, connentor):

    connection = connentor()
    cursor = connection.cursor()

    for item in video_data_list:

        video_id = item['id'].strip()
        title = item['snippet']['title'].strip()
        time = item['snippet']['publishedAt']
        views = item['statistics'].get('viewCount', 0)
        likes = item['statistics'].get('likeCount', 0)
        dislikes = item['statistics'].get('dislikeCount', 0)
        comments_cnt = item['statistics'].get('commentCount', 0)
        comments = item['replies']
        query = query.strip()
        # commnets는 댓글 여러개가 담긴 리스트형태이다.
        # 따라서 반복문을 통해서 strip()을 날려야한다.
        comments = [one_reply.strip() for one_reply in comments]

        video_data = {
            'video_id': video_id,
            'title': title,
            'time': time,
            'views': views,
            'likes': likes,
            'dislikes': dislikes,
            'comments_cnt': comments_cnt,
            'comments': comments,
            'query': query
        }

        insert_query = """
        INSERT INTO video_datas (video_id, title, time, views, likes, dislikes, comments_cnt, comments, query)
        VALUES (%(video_id)s, %(title)s, %(time)s, %(views)s, %(likes)s, %(dislikes)s, %(comments_cnt)s, %(comments)s, %(query)s);
        """

        cursor.execute(insert_query, video_data)

    connection.commit()
    cursor.close()
    connection.close()


# 채널명 조사 및 DB 에 채널명과 채널id 삽입
def inputChannels(channel_list, query):
    connection = get_db_connection()
    cursor = connection.cursor()

    for item in channel_list:
        name = item['name']
        id = item['id']
        subscribers = item['subscribers']
        topic = query

        channel_datas = {
            "name": name,
            "id": id,
            "subscribers": subscribers,
            "topic": topic
        }

        insert_query = """
        INSERT INTO channels (name, id, subscribers, topic)
        VALUES (%(name)s, %(id)s, %(subscribers)s, %(topic)s);
        """
        cursor.execute(insert_query, channel_datas)

    connection.commit()
    cursor.close()
    connection.close()
