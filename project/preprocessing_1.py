from DB_retriever import select_query
import pandas as pd
import re
from pykospacing.kospacing import Spacing
from DB_connector import get_db_connection
# 띄어쓰기 정규화
spacing = Spacing()

# rows = select_query('comments')


# # 각 비디오당, 가지고 있는 댓글 개수 알아보기
# sum = 0
# for row in rows:
# row[0] 에서 굳이 인덱스0을 붙이는 이유는 밖의 튜플 하나를 없애기 위해서이다. row는 현재는, 원래 원소 하나값만 지금 들어있음.
#     length = len(row[0].replace('{', '').replace('}', '').split('",'))
#     sum += length


# for row in rows:
#     print(row[0].replace('{', '').replace('}', '').split('",'))


# 비디오 제목과 댓글 같이 불러오기
# 가져와서 데이터 프레임으로 바꿔야 한다.
rows = select_query(['title', 'comments'])
# 댓글은 문자열 형태이다...
# 제목도 문자열 형태이다.
# 영상당 하나는 튜플 형태이다. 그러니깐 튜플 하나에 문자열 형태 2개의 원소가 존재한다.


# DB에 저장되어 있는 데이터 가져다가 딕셔너리 형태로 만들기
data_list = []
for tuple in rows:
    title = tuple[0]
    comment_list = tuple[1].replace("{", "").replace("}", "").split('","')
    comment_list[0] = comment_list[0].replace('"', '')

    dic = {}
    dic['title'] = title
    dic['comments'] = comment_list

    data_list.append(dic)


# <<<<<<<<<클렌징>>>>>>>>>>>>
# 1. 특수문자, 외국어 제거


# 정규식을 이용해서 한글, 공백(띄어쓰기 한칸)를 제외하고 모두 제거하기
def clean_text(text):
    cleaned_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", text)
    cleaned_text = re.sub("\s+", " ", cleaned_text)  # 연속된 공백을 하나로 축소
    return cleaned_text

# 위 정규식을 거치고 나면 텅텅 비어 있는 값 '', 혹은 공백만 있는 값들이 있을 수 있음
# 이 값들도 제거하는 정규식 => '^\s*$'


# 반복문으로 데이터 정리
for data in data_list:
    data['title'] = clean_text(data['title'])
    if re.match(r'^\s*$', data['title']):
        data['title'] = ''

    for i, comment in enumerate(data['comments']):
        data['comments'][i] = clean_text(comment).strip()

# 48번 인덱스는 모두 타이틀과 댓글이 비어있어서 봤더니 사실 외국 채널이었음 모든것이 러시아 문자라서 빈값 '' 으로 나왔던 거였음.
# print(data_list[48])

# 띄어쓰기 정규화하고 토큰화 진행
for data in data_list:
    data['title'] = spacing(data['title'])
    for i, comment in enumerate(data['comments']):
        data['comments'][i] = spacing(comment)
        print(data['comments'][i])


# CREATE TABLE video_datas_normalized (
#     id SERIAL PRIMARY KEY,
#     title TEXT,
#     comments TEXT
# );

conn = get_db_connection()
cursor = conn.cursor()

for data in data_list:
    print(f'{data} 삽입 할 차례 입니다.')
    cursor.execute(
        "INSERT INTO video_datas_normalized_1 (title, comments) VALUES (%s, %s)",
        (data["title"], data["comments"])
    )

conn.commit()
cursor.close()
conn.close()


# 위에서 만든 딕셔너리 형태를 다시 판다스 데이터프레임으로 전환
# df = pd.DataFrame(data_list)


# 기본 정보 확인
# print(df.info())

# 널값 확인
# print(df.isnull().sum())


# df['title'] = df['title'].apply(spacing)
# print(df['title'])


# gpt를 이용한 라벨링 작업
