# from random import shuffle
# list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# shuffle(list)
# chicken  = list[0]
# coffee = list[1:4]
# print (f"Today's lucky chicken is number {chicken}")
# print("Today's coffee coupons go to numbers {0}, {1}, and {2}.".format(coffee[0], coffee[1], coffee[2]))

# lst = []
# for i in range(1,21):
#     lst.append(i)
# shuffle(lst)
# print(f"Today's lucky chiken is for {lst[0]}")
# print(f"Today's coffee coupons go to {lst[1]}, {lst[2]}, and {lst[3]}")

# for i in range(4):
#     print(i)

# # skipped if, while, for

# absent =[2,5]
# for student in range(1,11):
#     if student in absent:
#         continue # skip the rest of the Loop ( print ~~ ) and go to the next iteration ( which is the next student number)
#     print("{0}, please read the book.".format(student))
#     print("reading is fun")

# no_book = [7]

# for i in range(11):
#     if i in no_book:
#         print("go to the library")
#         break

from random import randint, random, randrange


# sts = [1,2,3,4,5]
# for i in range(len(sts)):
#     sts[i] += 100
#     print(sts[i])

# stss = ["cy", "jw", "yw"]
# stss = [len(i) for i in stss]
# print(stss)

# def randomo(x):
#     return random() * x


# passenger = [i for i in range(51)]

# for i in passenger:
#     if i == x:
#         print("You are lucky today! You get a free ride.")
#     else:
#         print("You have to pay for the ride.")

# cnt = 0
# for i in range(51):
#     time = randrange(5,51)
#     if time >= 5 and  time <= 15:
#         print(f"[0] {i}th passenger: {time} minutes, got to ride")
#         cnt += 1
#     else: 
#         print(f"[ ] {i}th passenger: {time} minutes, cannot ride") 
# print(f"Total {cnt} passengers got to ride.")


# def profile(name="cy", age, main_lang):
#     print(f"name: {name}\tage: {age}\tmain language: {main_lang}")
# profile("cy", 20, "python")
# profile("jw", 25, "java")
# profile("yw", 22, "c++")
# profile(name="cy", main_lang="python", age=20)

# def profile2(name, age, *language):
#     print(f"name : {name }\tage: {age}\t", end=" ")
#     for lang in language:
#         print(lang, end=" ")
y = input("enter your sex (male or female)")
x = input("enter your height in meters:")
def calculate(y, x):
    if  y == "male":
        weight = x**2*22
    else:
        weight = x**2*21
    return weight
print("Your ideal weight is {0:.2f} kg.".format(calculate(y, float(x)/100)))