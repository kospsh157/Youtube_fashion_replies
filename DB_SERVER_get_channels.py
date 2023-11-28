from DB_SERVER_connector import connentor


def get_channels():
    conn, cursor = connentor()
    query = 'select id from channels'
    cursor.execute(query)
    channels_ids = cursor.fetchall()
    cursor.close()
    conn.close()

    # fetchall()함수로 받아올 경우 하나의 인스턴스를 튜플 하나로 감싸서, 튜플 리스트로 받아온다.
    channels_list = [id[0] for id in channels_ids]

    return channels_list
