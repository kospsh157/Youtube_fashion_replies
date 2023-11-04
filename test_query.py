from DB_retriever import select_query


rows = select_query(['comments'])

# '","' 기준으로 split

# 모든 댓글 개수 알아내기
total_comments_cnt = 0
for i in range(len(rows)):
    split_data = rows[i][0].split('","')
    total_comments_cnt += len(split_data)


print(f"총 댓글의 개수는 {total_comments_cnt}개 입니다.")

# 결과의 길이를 반환
