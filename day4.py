# class Buildngunit(unit):
#     def __init__(self, name, hp, location):
#         super().__init__(name, hp, 0)
#         self.location = location

# class Unit:
#     def __init__(self):
#         print("생성자")
# class flyable:
#     def __init__(self):
#         print("flyable 생성자")
        
# class flaybleunit(Unit, flyable): # 상속받는 클래스는 괄호안에 반드시 작성 해야함,
#     # 그리고 순서가 super 에 영향을 미침 + 클래스이름.__init__(self, 여기에 그 클래스 안에있는 모든 정보를 넣어줘야함) 
#     def __init__ (self):
#         #super().__init__()
#         Unit.__init__(self)
#         flyable.__init__(self)
# cksdud = flaybleunit()



try:
    print("나누기 전용")
    num1 = int(input("첫 번째 숫자를 입력하세요: "))
    num2 = int(input("두 번째 숫자를 입력하세요: "))
    print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))
except ValueError:
    print("에러! 잘못된 값을 입력하였습니다.")