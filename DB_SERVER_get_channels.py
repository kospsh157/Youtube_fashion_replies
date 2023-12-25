from DB_SERVER_connector import DB_server_connentor


# 집 로컬 db 주소
'''
ip: 192.168.0.101:5432 
주피터: 8888
ssh: 22
'''

# api서버 주소
'''
ip: 192.168.0.201
포트들:
    리액트 개발: 3000
    웹: 80
    ssh: 222

'''


def get_channels():
    conn, cursor = DB_server_connentor()
    query = 'select id from channels'
    cursor.execute(query)
    channels_ids = cursor.fetchall()
    cursor.close()
    conn.close()

    # fetchall()함수로 받아올 경우 하나의 인스턴스를 튜플 하나로 감싸서, 튜플 리스트로 받아온다.
    channels_list = [id[0] for id in channels_ids]

    print(channels_list)
    return channels_list


get_channels()