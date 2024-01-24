

# 시각화
# 데이터 조회
print(df.head())  # 데이터프레임의 상위 5개 행 조회


# # 결측치 처리
# df['Age'] = df['Age'].fillna(value=0)  # 'Age' 컬럼의 결측치를 0으로 대체

# # 데이터 타입 변환
# df['Age'] = df['Age'].astype(int)  # 'Age' 컬럼을 정수형으로 변환

# # 데이터 필터링
# filtered_df = df[df['Age'] > 30]  # 'Age'가 30보다 큰 행만 필터링

# # 데이터 정렬
# sorted_df = df.sort_values(by='Age', ascending=False)  # 'Age' 기준으로 내림차순 정렬

# # 그룹화 및 집계
# grouped_df = df.groupby('City').mean()  # 'City'를 기준으로 그룹화하고 평균을 계산


# # 문자열 처리
# df['City'] = df['City'].str.lower()  # 'City' 컬럼의 문자열을 소문자로 변환

# # 조인
# other_data = {
#     'Name': ['Alice', 'Bob'],
#     'Salary': [70000, 80000]
# }
# other_df = pd.DataFrame(other_data)
# merged_df = pd.merge(df, other_df, on='Name')  # 'Name'을 기준으로 df와 other_df 병합

# # 피벗 테이블
# # 'City'를 인덱스로 하여 'Age'의 평균을 계산
# pivot_table = df.pivot_table(values='Age', index='City', aggfunc='mean')

# # 데이터프레임 출력
# print(merged_df)
