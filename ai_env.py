import numpy as np

# -------------------------
# 코너와 엣지 번호 매기기
# -------------------------
corner_index = {'UFR':0, 'URB':1, 'UBL':2, 'ULF':3,
                'DFR':4, 'DRB':5, 'DBL':6, 'DLF':7}
edge_index = {'UF':0, 'UR':1, 'UB':2, 'UL':3,
              'FR':4, 'RB':5, 'BL':6, 'LF':7,
              'DF':8, 'DR':9, 'DB':10, 'DL':11}

# 초기 퍼뮤테이션과 오리엔테이션
corner_permu = [0,1,2,3,4,5,6,7]
corner_orient = [0]*8
edge_permu = list(range(12))
edge_orient = [0]*12

# -------------------------
# Rubiks 클래스 정의
# -------------------------
class Rubiks:
    def __init__(self):
        self.corner_permu = corner_permu.copy()
        self.corner_orient = corner_orient.copy()
        self.edge_permu = edge_permu.copy()
        self.edge_orient = edge_orient.copy()
    
    # ---------------------
    # 기본 무브 정의 예시
    # ---------------------
    def R(self):
        # 코너 퍼뮤테이션
        self.corner_permu[0], self.corner_permu[1], self.corner_permu[5], self.corner_permu[4] = \
            self.corner_permu[4], self.corner_permu[0], self.corner_permu[1], self.corner_permu[5]
        # 코너 오리엔테이션
        for i in [0,1,4,5]:
            self.corner_orient[i] = (self.corner_orient[i] + (2 if i in [0,5] else 1)) % 3
        # 엣지 퍼뮤테이션
        self.edge_permu[1], self.edge_permu[4], self.edge_permu[9], self.edge_permu[5] = \
            self.edge_permu[4], self.edge_permu[9], self.edge_permu[5], self.edge_permu[1]
        # 엣지 오리엔테이션 (R move는 엣지 오리엔테이션 변화 없음)

    def R_prime(self):
        # R의 반대 방향
        for _ in range(3):  # R' = R 세 번
            self.R()
    
    def R2(self):
        for _ in range(2):
            self.R()
    
    # ---------------------
    # 나머지 무브 정의
    # ---------------------
    # L, L', L2
    def L(self):
        self.corner_permu[2], self.corner_permu[3], self.corner_permu[7], self.corner_permu[6] = \
            self.corner_permu[3], self.corner_permu[7], self.corner_permu[6], self.corner_permu[2]
        for i in [2,3,6,7]:
            self.corner_orient[i] = (self.corner_orient[i] + (2 if i in [3,6] else 1)) % 3
        self.edge_permu[3], self.edge_permu[7], self.edge_permu[11], self.edge_permu[6] = \
            self.edge_permu[7], self.edge_permu[11], self.edge_permu[6], self.edge_permu[3]
    
    def L_prime(self):
        for _ in range(3):
            self.L()
    
    def L2(self):
        for _ in range(2):
            self.L()
    
    # U, U', U2
    def U(self):
        self.corner_permu[0], self.corner_permu[3], self.corner_permu[2], self.corner_permu[1] = \
            self.corner_permu[3], self.corner_permu[2], self.corner_permu[1], self.corner_permu[0]
        self.edge_permu[0], self.edge_permu[3], self.edge_permu[2], self.edge_permu[1] = \
            self.edge_permu[3], self.edge_permu[2], self.edge_permu[1], self.edge_permu[0]
    
    def U_prime(self):
        for _ in range(3):
            self.U()
    
    def U2(self):
        for _ in range(2):
            self.U()
    
    # D, D', D2
    def D(self):
        self.corner_permu[4], self.corner_permu[5], self.corner_permu[6], self.corner_permu[7] = \
            self.corner_permu[5], self.corner_permu[6], self.corner_permu[7], self.corner_permu[4]
        self.edge_permu[8], self.edge_permu[9], self.edge_permu[10], self.edge_permu[11] = \
            self.edge_permu[9], self.edge_permu[10], self.edge_permu[11], self.edge_permu[8]
    
    def D_prime(self):
        for _ in range(3):
            self.D()
    
    def D2(self):
        for _ in range(2):
            self.D()
    
    # F, F', F2
    def F(self):
        self.corner_permu[0], self.corner_permu[4], self.corner_permu[7], self.corner_permu[3] = \
            self.corner_permu[3], self.corner_permu[0], self.corner_permu[4], self.corner_permu[7]
        for i in [0,3,4,7]:
            self.corner_orient[i] = (self.corner_orient[i] + 1) % 3
        self.edge_permu[0], self.edge_permu[4], self.edge_permu[8], self.edge_permu[7] = \
            self.edge_permu[7], self.edge_permu[0], self.edge_permu[4], self.edge_permu[8]
    
    def F_prime(self):
        for _ in range(3):
            self.F()
    
    def F2(self):
        for _ in range(2):
            self.F()
    
    # B, B', B2
    def B(self):
        self.corner_permu[1], self.corner_permu[2], self.corner_permu[6], self.corner_permu[5] = \
            self.corner_permu[5], self.corner_permu[1], self.corner_permu[2], self.corner_permu[6]
        for i in [1,2,5,6]:
            self.corner_orient[i] = (self.corner_orient[i] + 1) % 3
        self.edge_permu[2], self.edge_permu[5], self.edge_permu[10], self.edge_permu[6] = \
            self.edge_permu[6], self.edge_permu[2], self.edge_permu[5], self.edge_permu[10]
    
    def B_prime(self):
        for _ in range(3):
            self.B()
    
    def B2(self):
        for _ in range(2):
            self.B()
