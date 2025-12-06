import random


nums = []
for i in range(10):
    x = random.randint(1, 100)
    nums.append(x)
def find_maximum(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

print(find_maximum(nums))
# 하나씩 번갈아서 비교해보면서 변수를 새로 저장


student = {"name": "cutie", "score": 94}
def grade_up_by_6(mapping: dict) -> dict:
    mapping["score"] += 6
    return mapping
    # for name, score in mapping.items():
    #     if name == "cutie" and score == 94:
    #         mapping[score] = mapping.values + 6
    # return mapping
  
print(grade_up_by_6(student))

def wprhq(x):
    return x**2
print(wprhq(5))