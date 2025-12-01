import random
# jwsgf = {
#     "Yedam": 1,
#     "Heewon": 2,
#     "online": 3,
#     "sharon": 4,
#     "fan": 5,
#     "eunchae": 6,
#     "naye": 7,
#     "wenjo": 8,
#     "bottari": 9,
#     "jua": 10,
# }

# def randomize_priorities(mapping: dict) -> dict:
#     """Assign a unique random priority 1..N to each name."""
#     names = list(mapping.keys())
#     n = len(names)
#     priorities = list(range(1, n + 1))
#     random.shuffle(priorities)
#     return {name: priorities[i] for i, name in enumerate(names)}

# def print_by_priority(mapping: dict) -> None:
#     """Print names ordered by ascending priority (1 is highest)."""
#     for name, pr in sorted(mapping.items(), key=lambda kv: kv[1]):
#         print(f"{pr:2d} - {name}")

# # Randomize priorities
# jwsgf = randomize_priorities(jwsgf)

# # Print in priority order
# print_by_priority(jwsgf)
    




# List = {"jinwoo":1, "yoonwoon":2, "chanyoung":3}

# def choose_boyfriend(mapping: dict) -> dict:
#     names = list(mapping.keys())
#     n = len(names)
#     priorities = list(range(1, n+1))
#     random.shuffle(names)
#     return {name: priorities[i] for i, name in enumerate(names)}
# def print_boyfriend(mapping: dict) -> None:
#     for name, pr in mapping.items():
#         print(f"{pr:2d} - {name}")

# List = choose_boyfriend(List)
# print_boyfriend(List)

# def bestmatch(mapping:dict, mapping2:dict) -> None:
#     # 진우의 value 와 jua 의 value 가 같으면 best match, in any case, return " no best match found"
#     found = False
#     for name, i in mapping.items():
#         for name2, j in mapping2.items(): 
#             if i == j and name == "jinwoo" and name2 == "jua":
#                 print("best match found")
#                 found = True
#                 break 
#         if found:
#             break
#     if not found:
#         print("no best match found")
                

# bestmatch(List, jwsgf)
  





# class kinda nervous


# class 찬영(name, hp, damage):
#     self.name = name
#     self.hp = hp
#     self,damage = damage 
#     def attack(self, location, damage)

# class 윤원(name, hp, damage):
#     self.name = name
#     self.hp = hp
#     self,damage = damage

# class 진우(name, hp, damage):
#     self.name = name
#     self.hp = hp
#     self,damage = damage
 
# cksdud =[ 찬영(i, 40, 20) for i in range (3)]
# ywsdud =[ 윤원(i, 40, 15) for i in range (3)]       
# jwsdud =[ 진우(i, 40, 25) for i in range (3)]

# 모든 캐릭터를 저장할 리스트
all_characters = []
class character:
    def __init__(self, name, hp, damage, isdead):
        self.name = name
        self.hp = hp
        self.damage = damage
        all_characters.append(self)  # 생성시 자동 추가
        self.isdead = False 
        
    def attack(self):
        # 자기 자신을 제외한 살아있는 캐릭터에서 랜덤 선택
        targets = [char for char in all_characters if char.name != self.name and not char.isdead]
        if targets:
            target = random.choice(targets)  # 객체를 선택
            print(f"{self.name} attacks {target.name} for {self.damage} damage!")
            target.hp -= self.damage
            print(f"{target.name}'s HP: {target.hp}")
            if target.hp <= 0:
                print(f"{target.name} has been defeated!")
                target.isdead = True
    


cksdud = character("찬영", 100, 20, False)
ywsdud = character("윤원", 100, 20, False)
jwsdud = character("진우", 100, 20, False)

print("\n=== 전투 시작! ===")

# 한 명이 죽을 때까지 반복
while not any(char.isdead for char in all_characters):
    x = random.randint(0, 2)  # 0, 1, 2 중 하나
    attacker = all_characters[x]
    if not attacker.isdead:
        attacker.attack()
print("\n=== 캐릭터 리스트 확인 ===")
print(f"첫 번째 캐릭터 이름: {all_characters[0].name}")
print(f"전체 캐릭터 수: {len(all_characters)}")
for i, char in enumerate(all_characters):
    print(f"{i}: {char.name} (HP: {char.hp}, Damage: {char.damage})")


