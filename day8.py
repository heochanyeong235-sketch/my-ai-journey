#numpy lesson
import numpy as np
array1 = np.array([1,2,3,4,5])
array2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
array3 = np.array([[[1,2,3,4,5],[6,7,8,9,10]],
                   [[11,12,13,14,15],[16,17,18,19,20]]])
print(array1)
print(array2)
print(array3)
print(array1.shape)
print(array2.shape)
print(array3.shape)
print(array1.ndim)
print(array2.ndim)
print(array3.ndim)

word = array3[0,0,0] + array2[0,0]
print(word)



array_a = np.array([[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16
                    ]])
print(array_a.shape)
print(array_a[0]) # 첫 번째 행 출력
print(array_a[:,2])  # 세 번째 열 출력
print(array_a[1:3]) # 두 번째와 세 번째 행 출력
print(array_a[:,1:3])  # 두 번째와 세 번째 열 출력
print(array_a[::-1])  # 행을 역순으로 출력
print(array_a[::3])  # 3행마다 출력
print(array_a[1:3, 1:3])  # 2x2 부분 배열 추출
print(array_a[2:4, 1:2])

# ==================== NumPy Shape (차원) 읽는 법 ====================
# Shape: (a, b, c) 형태로 표현됨
#
# 📌 규칙: 뒤에서부터 읽는다! (오른쪽 → 왼쪽)
#
#    (a, b, c) 의 의미:
#    - c: 가장 안쪽 차원 = 열(Column) = 가로줄 개수 = ROW의 원소 개수
#    - b: 중간 차원 = 행(Row) = 세로줄 개수 = 한 면의 행 개수
#    - a: 가장 바깥 차원 = 깊이(Depth) = 면(Layer) 개수 = 2D 배열이 몇 개인지
#
# 🔍 예제:
#    Shape (2, 3, 4) 의미:
#    - 4: 각 행에 숫자 4개 (가로로 4칸)
#    - 3: 각 면에 행이 3개 (세로로 3줄)
#    - 2: 이런 면이 2개 (2개의 2D 배열)
#
#    실제 배열:
#    [[[1, 2, 3, 4],      ← 1번째 면 (3행 × 4열)
#      [5, 6, 7, 8],
#      [9,10,11,12]],
#
#     [[13,14,15,16],     ← 2번째 면 (3행 × 4열)
#      [17,18,19,20],
#      [21,22,23,24]]]
#
# 📌 슬라이싱 규칙:
#    array_a[행, 열]
#    - : (콜론) = 해당 차원 전체 선택
#    - array_a[:,2] = 모든 행의 3번째 열 (인덱스 2)
#    - array_a[1:3] = 2번째~3번째 행
#    - array_a[::2] = 2칸씩 건너뛰며 선택
# ==================================================================== 
 #vectorize math ofunctions

radii = np.array([1,2,4])
print(np.pi * radii**2)
print(np.sqrt(radii))# 제곱근
print(np.log(radii))# 자연로그
print(np.exp(radii))# 지수 함수
print(np.floor(radii)) # 내림
print(np.ceil(radii)) # 올림)

scores = np.array([88.5, 92.3, 79.8, 85.0, 90.2])
print(scores == 100) # 각 요소가 100인지 비교
print(scores >= 90)  # 각 요소가 90보다 큰지 비교
scores[scores < 60] = 60  # 60 미만인 점수를 60으로 설정, curve 최저점수를 curve



array_a1 = np.array([1,2,3,4])
array_a2 = np.array([[1],[2],[3],[4]])
print(array_a1.shape)  # (1,4) == (4,)
print(array_a2.shape)  # (4, 1)

# ==================== 브로드캐스팅 (Broadcasting) 룰 ====================
# 브로드캐스팅: 크기가 다른 배열끼리 연산할 때 자동으로 shape을 맞춰주는 것
#
# 📌 룰 1: 뒤에서부터 차원을 비교한다
#    예: (3, 4) 와 (4,) 비교 → 마지막 차원 4가 같으니까 OK
#
# 📌 룰 2: 차원이 1이거나 없으면 자동으로 늘려서 맞춘다
#    예: (4,) → (1, 4) → (4, 4)로 복사됨
#        (4, 1) → (4, 4)로 복사됨
#
# 📌 룰 3: 크기가 다르고 1도 아니면 에러!
#    예: (3,) 와 (4,) → 에러! (3과 4가 맞지 않음)
#
# 🔍 아래 예제 분석:
#    array_a1 shape: (4,)   → 브로드캐스팅 시 (1, 4)로 취급
#    array_a2 shape: (4, 1)
#    연산: (4, 1) * (1, 4) → 둘 다 (4, 4)로 확장됨
#    결과:
#    [[1*1, 1*2, 1*3, 1*4],     [[1,  2,  3,  4],
#     [2*1, 2*2, 2*3, 2*4],  →   [2,  4,  6,  8],
#     [3*1, 3*2, 3*3, 3*4],      [3,  6,  9, 12],
#     [4*1, 4*2, 4*3, 4*4]]      [4,  8, 12, 16]]
# ========================================================================

print(array_a1 * array_a2)
print(np.std(array_a1)) # 표준편차
print(np.var(array_a1)) # 분산
print(np.mean(array_a1)) # 평균
print(np.median(array_a1)) # 중앙값
print(np.sum(array_a1))# 합계
print(np.min(array_a1)) # 최소값
print(np.max(array_a1)) # 최대값
print(np.argmin(array_a)) # 최소값 인덱스 what is the index of the minimum value

print(np.argmax(array_a)) # 최대값 인덱스 what is the index of the maximum value 

# axis = 0 은 열 방향으로 연산 (각 열의 합계), 열은 세로 방향
print(np.sum(array_a, axis=0))
# axis = 1 은 행 방향으로 연산 (각 행의 합계), 행은 가로 방향
print(np.sum(array_a, axis=1))
# axis = None 은 전체 요소에 대해 연산
print(np.sum(array_a, axis=None))
# axis = 2 은 3차원 배열에서 깊이 방향으로 연산 (각 면의 합계)
array_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(np.sum(array_3d, axis=2)) # result = [[ 3  7] [11 15]] 


#filtering with numpy
ages = np.array([[22, 25, 18, 30, 27, 19, 24],[29, 21, 23, 26, 28, 20, 31]])
teenagers = ages[ages < 20]
adults = ages[(ages >= 20) & (ages < 65)]

print(teenagers)


adults = np.where(ages >= 18, ages, 0) 


# 변수
rng = np.random.default_rng(0)
print(rng.integers(1,7))

print(rng.integers(low=1, high=7, size=(3,4)))
print(rng.uniform(low=1, high=1, size=3))


rng_1 = np.random.default_rng() # 이름은 rng_11 뭐사기든 상관 ㄴ but defult_rng() 이렇게는 꼭 써야함
array_11= np.array([1,2,3,4,5])
rng_1.shuffle(array_11)
print(array_11)

fruits = np.array(['apple', 'banana', 'cherry', 'date'])
print(rng_1.choice(fruits))
print(rng_1.choice(fruits, size=2, replace=False)) # 중복 없이 2개 선택