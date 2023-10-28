from DB_retriever import select_query

rows = select_query('comments')


# 각 비디오당, 가지고 있는 댓글 개수 알아보기
sum = 0
for row in rows:
    length = len(row[0].replace('{', '').replace('}', '').split('",'))
    sum += length


# 총 댓글의 개수
print(sum)


# 비디오 제목과 댓글 같이 불러오기
# 가져와서 데이터 프레임으로 바꿔야 한다.
