print(5)
print(5>10)
print(True)
print(True and False)

animal = "dog"
name = "cy"
age = 4
hobby = "walking"
is_audult = age >=3

print("My animal is " + animal + "and its name is " + name)
print(name + " is " + str(age) + " years old")
print(name + " is adult? " + str(is_audult))



station = "sadang"
print(station + "행 열차가 들어오고 있습니다.")

print(2**3) # 제곱
print(5%3) # 나머지
print(10//3) # 몫
print(10>3) # True

print(abs(-5)) # 절댓값
print(pow(4,2)) # 4^2
print(max(5,12)) # 최대값
print(min(5,12)) # 최소값
print(round(3.14)) # 반올림

from math import *
print(floor(4.99)) # 내림
print(ceil(4.01)) # 올림
print(sqrt(16)) # 제곱근

from random import *
print(random()) # 0.0 ~ 1.0 미만의 임의의 값 생성
print(random() * 10) # 0.0 ~ 10.0 미만의 임의의 값 생성
print(int(random() * 10)) # 0 ~ 10 미만의 임의의 값 생성
print(int(random() * 10) + 1) # 1 ~ 10 이하의 임의의 값 생성
print(randrange(1,11)) # 1 ~ 10 이하의 임의의 값 생성
print(randint(1,10)) # 1 ~ 10 이하의 임의의 값 생성

x = randint(4,28) # inculding both starting and ending value 
print(f"the offline study day is on the {x}th of this month.")

sentence = 'I am a guy'
print(sentence)
sentence2 = "Python is fun"
print(sentence2)
sentence3 = """나는 소년이고 나는 바보다 
나는 그런줄 알았는데 아니었다
나는ㄴ 글쎄다 잘 모르겠다 """
print(sentence3)

chanyoung = "070430-1234567"

print(f"성별 : {chanyoung[2]}")

python = "Python is Amazing"
print(python.lower())
print(python.upper())
print(python[0].islower())
print(len(python))
print(python.replace("Python", "Java"))
index = python.index("n")
index = python.index("n", index + 1) # 두번째 n의 위치
print(python.count("m"))

New_Python = list(python) # string "Python is amazing" 을 list 로써 변환 예를들면, P = New_python[0] 이런식으로
index = python.index("n")
New_Python[index] = "1" 
print("".join(New_Python))
print(New_Python[0:3])

y = "p"
print("나는 %d살 입니다." % 19) # %d 는 정수 만 
print("나는 %s을 좋아해요." % y) # %s 는 string
print ("나는 %c 로 시작해요" % y)

jinwoo = "jinwoo likes both {1} and {0}" .format("apple", "banana")
print(jinwoo) 

print("백문이 불여일견 \n 백견이 불여일타")

menu =("돈까스", "치즈까스")
print(menu[0])
name = " jinwoo" 
age = 25
hobby = "coding"    
print(f"나는 {name}이고, 나이는 {age}살이며, 취미는 {hobby}입니다.")
      
my_set = {1,2,3,3,3}
print(my_set)

java = {"A", "B", "C"}
python = set (["A", "B", "D"])
print(java & python) # 교집합
print(java.intersection(python)) # 교집합])