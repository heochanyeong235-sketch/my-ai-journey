# list =[]
# for i in range(0,5):
#     x = int(input("Enter a number:"))
#     list.append(x)
# def colaz_function(list):
#     while any(list) != 1:
#         for i in list:
#             if i % 2 == 0:
#                 i = i // 2
#             elif i % 2 == 1:
#                 i = 3 * i + 1
#     return list
# def returning(:
#     for i in list:
#         if list[i] != 1:
#             print(False)
#         if list[i] == 1:
#             print (True)
        
# colaz_function(list)
# returning(list)
import random
# x = int(input("Enter a number:"))

# def colaz_function(x):
#     while x != 1:
#         if x % 2 == 0:
#             x = x // 2
#             print(x)
#         elif x % 2 == 1:
#             x = 3 * x + 1
#             print(x)
#     if x == 1:
#         return True
#     else: 
#         return False
#     return 

# print(colaz_function(x))

numbers = []
for i in range(1, 11):
    x = random.randint(1, 1000)
    numbers.append(x)

print(f"시작 숫자들: {numbers}")

def multiple_numbers(nums):
    result = []
    # 각 숫자를 1이 될 때까지 계산
    for num in nums:
        while num != 1:
            if num % 2 == 0:
                num = num // 2
            else:  # num % 2 == 1
                num = 3 * num + 1
        # 1이 되면 True 추가
        result.append(True)
    return result
def ifalltrue(lst):
    if all(lst):
        x = print("모든 숫자가 1에 도달했습니다.")
    return x
print(f"결과: {multiple_numbers(numbers)}")
print(ifalltrue(multiple_numbers(numbers)))







def prime_number(num_range):
    prime_number =[]
    not_prime_number = []
    for num in range(2,100):
        for i in range(2,num):
            if num % i == 0:
                not_prime_number.append(num)
                break
        else:
            prime_number.append(num)
    return prime_number
print(prime_number(range(1,100)))


# import tkinter as tk
# #tk 는 tkinter 모듈의 약어  즉 tkinter 모듈을 tk 라는 이름으로 사용하겠다는 의미
# window = tk.Tk()
# window.title("My GUI Application")
# window.geometry("1000x1000")


# label = tk.Label(window, text="hello!", font =("Times New Roman", 30))
# label.pack()
# #Pack 은 레이블을 창에 추가하는 함수
# def buttom_click():
#     label.config(text="버튼이 클릭되었습니다!") # label.config 는 label 의 속성을 변경하는 함수 즉 텍스트를 변경하는 것
# Label = tk.Label(window, text="click the button", font=("times new roman", 20))
# Label.pack()
# button = tk.Button(window, text="click me", command=buttom_click, font=("Arial", 15))
# #command 는 버튼이 클릭되었을 때 호출되는 함수 지정
# button.pack()
# window.mainloop()



# ===== 콜라츠 추측 시각화 =====
import tkinter as tk
import random
import math

def colaz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

def draw_colaz_sequence(start_num):
    sequence = colaz_sequence(start_num)
    
    window = tk.Tk()
    window.title(f"Collatz Sequence: {start_num}")
    canvas = tk.Canvas(window, width=1000, height=700, bg="white")
    canvas.pack()

    # 제목
    canvas.create_text(500, 30, text=f"Collatz Sequence starting from {start_num}", #canvas create_text 는 캔버스에 텍스트를 그리는 "함수"
                      font=("Arial", 16, "bold"), fill="black")
    canvas.create_text(500, 55, text=f"Steps: {len(sequence)}, Max: {max(sequence)}", 
                      font=("Arial", 12), fill="gray")

    max_value = max(sequence)
    scale_x = 900 / len(sequence)
    scale_y = 550 / max_value

    # 그래프 그리기
    for i in range(len(sequence) - 1):
        x1 = i * scale_x + 50
        y1 = 650 - sequence[i] * scale_y
        x2 = (i + 1) * scale_x + 50
        y2 = 650 - sequence[i + 1] * scale_y
        
        # 그라데이션 효과
        color_intensity = int(255 * (1 - i / len(sequence)))
        color = f'#{color_intensity:02x}{100:02x}{255:02x}'
        canvas.create_line(x1, y1, x2, y2, fill=color, width=2) # canvas create_line 는 캔버스에 선을 그리는 함수

    # 축 그리기
    canvas.create_line(50, 650, 950, 650, fill="black", width=1)  # x축, 저 숫자들은 어디서 부터 어디까지 그릴지 좌표 더 정확하게는 (x1, y1, x2, y2) 그리고 
    canvas.create_line(50, 100, 50, 650, fill="black", width=1)   # y축

    window.mainloop()


# ===== 소수 분포 시각화 (리만 가설 관련) =====
def get_primes(limit):
    """에라토스테네스의 체로 소수 찾기"""
    sieve = [True] * limit # limit 크기의 리스트를 True 로 초기화, sieve 는 체 라는 뜻, 체 는 소수를 찾는 알고리즘 이름
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit, i):
                sieve[j] = False
    
    return [i for i in range(limit) if sieve[i]]

def prime_counting_function(n, primes):
    """π(n): n 이하의 소수 개수"""
    return sum(1 for p in primes if p <= n)

def draw_prime_distribution(limit=1000):
    primes = get_primes(limit)
    
    window = tk.Tk()
    window.title("Prime Number Distribution (Riemann Hypothesis)")
    canvas = tk.Canvas(window, width=1000, height=700, bg="white")
    canvas.pack()

    # 제목
    canvas.create_text(500, 30, text="Prime Distribution & li(x) approximation", 
                      font=("Arial", 16, "bold"), fill="black")
    canvas.create_text(500, 55, text=f"Primes up to {limit}: {len(primes)}", 
                      font=("Arial", 12), fill="gray")

    # 스케일 설정
    scale_x = 900 / limit
    scale_y = 550 / len(primes)

    # π(x) 그래프 (실제 소수 개수)
    for i in range(2, limit, 5):
        count = prime_counting_function(i, primes)
        x = i * scale_x + 50
        y = 650 - count * scale_y
        canvas.create_oval(x-1, y-1, x+1, y+1, fill="blue", outline="blue") # canvas create_oval 는 타원을 그리는 함수 즉 여기서는 점을 찍는 용도로 사용

    # x/ln(x) 근사 그래프 (소수 정리)
    for i in range(2, limit, 10):
        approx = i / math.log(i)
        x = i * scale_x + 50
        y = 650 - approx * scale_y
        canvas.create_oval(x-1, y-1, x+1, y+1, fill="red", outline="red")

    # 범례
    canvas.create_line(100, 680, 140, 680, fill="blue", width=2)
    canvas.create_text(180, 680, text="π(x) (actual prime count)", 
                      font=("Arial", 10), fill="blue", anchor="w")
    
    canvas.create_line(450, 680, 490, 680, fill="red", width=2)
    canvas.create_text(530, 680, text="x/ln(x) (approximation)", 
                      font=("Arial", 10), fill="red", anchor="w")

    # 축
    canvas.create_line(50, 650, 950, 650, fill="black", width=1)
    canvas.create_line(50, 100, 50, 650, fill="black", width=1)

    window.mainloop()


# ===== 실행 =====
if __name__ == "__main__":
    print("1. 콜라츠 추측 시각화")
    print("2. 소수 분포 시각화")
    choice = input("선택 (1 or 2): ")
    
    if choice == "1":
        num = int(input("시작 숫자를 입력하세요 (예: 27): "))
        draw_colaz_sequence(num)
    elif choice == "2":
        limit = int(input("범위를 입력하세요 (예: 1000): "))
        draw_prime_distribution(limit)
    else:
        print("잘못된 선택입니다.")





