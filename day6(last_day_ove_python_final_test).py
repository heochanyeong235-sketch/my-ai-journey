# # list
# list = [i for i in range(1,101)]
# def nfun(lst):
#     result = []
#     for i in lst:
#         if i % 3 == 0:
#             result.append(i)
#     return result
# print(nfun(list))

# #ditionary
# dict = {i: i for i in range(1, 101)}
# dict1 = {'a': 53, 'b':34, 'c':56, 'd':12, 'e':78, 'f':23, 'g':89}
# def find_i(dct):
#     result = {}
#     for key, value in dct.items():
#         if value > 50:
#             result[key] = value
#     return result
# print(find_i(dict1))

# # simper
# dict2 = {i: i for i in range(1, 101)}
# new_dict = {key: value for key, value in dict2.items() if value > 50}
# print(new_dict)

# # 글로벌 변수 바꾸는 코드
# count = 0 
# def increment():
#     global count
#     count += 1 
#     print(f"count: {count}")
# increment()
# increment()
# increment()
# print(f"Final count: {count}")

# #labda + map 으로 리스트의 모든 숫자 제곱하기 
# # def sqaure(x):
#     #return x**2
# #lambda x: x**2
# numbers = [1, 2, 3, 4, 5]
# squared = map(lambda x: x**2, numbers)# map 은 각각의 요소에 함수를 적용
# print(squared)

# #class 하나에 __init__ 과 __str__ 메서드 만들기
# class sibal:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#     def __str__(self): # str 함수는 객체를 문자열로 표현할 때 호출됨 즉 print 함수로 객체를 출력할 때 자동으로 호출
#         return f"Name: {self.name}, Age: {self.age}"

# # try except else finally 구조 완성하기 
# x1= float(input("Enter a number: "))
# x2 = float(input("Enter another number: "))
# def devide_numbers(a,b):
#     try: 
#         return a / b
#     except ZeroDivisionError:
#         return "Error: Division by zero is not allowed."
#     else:
#         print("Division performed successfully.")
#     finally:
#         print("Execution of devide_numbers is complete.")
# print(devide_numbers(x1, x2))

# # 제너레이터 함수 하나 만들기 (yeild 사용) 
# # yeild 는 함수의 실행을 일시 중지하고 값을 반환한 다음, 다시 호출되면 중지된 위치에서 계속 실행을 재개
# # ex)
# def countdown(n):
#     print("Countdown starts!")
#     while n > 0:
#         yield n
#         n -= 1
#     print("Countdown finished!")


# dict_of_loved = { "jua": 2020, "sua": 2021, "boa": 2022, "yoona": 2023, "hanna": 2024}

# def jinwoogfs(mapping: dict):
#     """진우가 사랑했던 그녀들을 순서대로 yield"""
#     print("진우가 사랑했던 그녀들 by year")
#     for key, value in mapping.items():
#         yield f"{key}: {value}"

# # 방법 1: for 루프로 직접 출력 (제너레이터 값을 받아서 출력)
# print("=== 제너레이터 사용 ===")
# for info in jinwoogfs(dict_of_loved):
#     print(info)

# #데코레이션 하나 직접 만들기 (@timer 같은거)
# import time
# @timer
# def timer1(func):
#     """함수 실행 시간 측정 데코레이터"""
#     def wrapper(*args, **kwargs): #*args, **kwargs 는 임의의 개수의 위치 인자와 키워드 인자를 받기 위해 사용 무슨말이냐면, 데코레이터가 적용된 함수가 어떤 인자를 받든지 간에 그 인자들을 그대로 전달할 수 있게 해줌
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
#         return result
#     return wrapper

# 파일 읽고 쓰기: 텍스트 파일에 "Hello, World!" 쓰고 다시 읽어서 출력
def file_read_write():
    filename = "hello.txt"
    # 파일 쓰기
    with open(filename, 'w') as f: # with and as 구문은 파일을 열고 자동으로 닫아주는 역할을 함, f 는 파일 객체
        f.write("Hello, World!")
    # 파일 읽기
    with open(filename, 'r') as f:
        content = f.read()
    print(content)
file_read_write()

# collections.Counter 써서 문자열에서 가장 많이 나온 알파벳 찾기. 
from collections import Counter
def most_common_letter(s):
    s = s.replace(" ", "").lower()  # 공백 제거 및 소문자 변환
    counter = Counter(s) # Counter 객체 생성, Counter 객체는 각 문자와 그 빈도수를 딕셔너리 형태로 저장
    most_common = counter.most_common(1) # most_common(n) 메서드는 가장 많이 나온 n개의 요소를 리스트 형태로 반환
    if most_common:
        letter, count = most_common[0] # letter과 count 를 둘다 쓴느 이유는 각각 가장 많이 나온 문자와 그 빈도수를 저장하기 위해서임
        return letter, count # count 는 리스트에 들어잇는 튜플의 두번째 요소임
    return None, 0
letter, count = most_common_letter("This is a sample string with several letters")
print(f"Most common letter: '{letter}' occurred {count} times.")