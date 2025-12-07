import numpy as np

color = {"w":1, "y":2, "g":3, "b":4, "o":5, "r":6}
Rubik_cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],
                       [[2,2,2],[2,2,2],[2,2,2]],
                       [[3,3,3],[3,3,3],[3,3,3]],
                       [[4,4,4],[4,4,4],[4,4,4]],
                       [[5,5,5],[5,5,5],[5,5,5]],
                       [[6,6,6],[6,6,6],[6,6,6]]])
print(Rubik_cube.shape)

# 전개도 형태로 나타내기 
def draw_cube_flat(color, Rubik_cube):
    print("전개도:")
    print()
    
    # 윗면 (면 0)
    for row in Rubik_cube[0]:
        # 윗면은 각 줄 앞에만 여백을 넣고, 마지막 줄 뒤에는 여백을 추가하지 않는다
        print("      " + ' '.join([list(color.keys())[list(color.values()).index(val)] for val in row]))

    # 중간줄 (왼쪽면, 앞면, 오른쪽면, 뒷면)
    for i in range(3):
        line = ""
        # 왼쪽면 (면 4)
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[4][i]])
        line += " "
        # 앞면 (면 2)
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[2][i]])
        line += " "
        # 오른쪽면 (면 5)
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[5][i]])
        line += " "
        # 뒷면 (면 3)
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[3][i]])
        print(line)
    
    
    # 아랫면 (면 1)
    print("      ", end="")
    for row in Rubik_cube[1]:
        print(' '.join([list(color.keys())[list(color.values()).index(val)] for val in row]))
        if row is not Rubik_cube[1][-1]:
            print("      ", end="")

class Moves:
    def __init__(self, Rubik_cube):
        self.cube = Rubik_cube
    def R(self):
        # 오른쪽 면 시계 방향 회전
        self.cube[5] = np.rot90(self.cube[5], -1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][:,2].copy()
        self.cube[0][:,2] = self.cube[2][:,2]
        self.cube[2][:,2] = self.cube[1][:,2]
        self.cube[1][:,2] = self.cube[3][:,0][::-1]
        self.cube[3][:,0] = temp[::-1]
    def R_prime(self):
        # 오른쪽 면 반시계 방향 회전
        self.cube[5] = np.rot90(self.cube[5], 1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][:,2].copy()
        self.cube[0][:,2] = self.cube[3][:,0][::-1]
        self.cube[3][:,0] = self.cube[1][:,2][::-1]
        self.cube[1][:,2] = self.cube[2][:,2]
        self.cube[2][:,2] = temp
    def R_double(self):
        # 오른쪽 면 180도 회전
        self.cube[5] = np.rot90(self.cube[5], 2)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][:,2].copy()
        self.cube[0][:,2] = self.cube[1][:,2]
        self.cube[1][:,2] = temp
        temp = self.cube[2][:,2].copy()
        self.cube[2][:,2] = self.cube[3][:,0][::-1]
        self.cube[3][:,0] = temp[::-1]
    def L(self):
        # 왼쪽 면 시계 방향 회전
        self.cube[4] = np.rot90(self.cube[4], -1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][:,0].copy()
        self.cube[0][:,0] = self.cube[3][:,2][::-1]
        self.cube[3][:,2] = self.cube[1][:,0][::-1]
        self.cube[1][:,0] = self.cube[2][:,0]
        self.cube[2][:,0] = temp
    def L_prime(self):
        # 왼쪽 면 반시계 방향 회전
        self.cube[4] = np.rot90(self.cube[4], 1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][:,0].copy()
        self.cube[0][:,0] = self.cube[2][:,0]
        self.cube[2][:,0] = self.cube[1][:,0][::-1]
        self.cube[1][:,0] = self.cube[3][:,2][::-1]
        self.cube[3][:,2] = temp
    def L_double(self):
        # 왼쪽 면 180도 회전
        self.cube[4] = np.rot90(self.cube[4], 2)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][:,0].copy()
        self.cube[0][:,0] = self.cube[1][:,0]
        self.cube[1][:,0] = temp
        temp = self.cube[2][:,0].copy()
        self.cube[2][:,0] = self.cube[3][:,2][::-1]
        self.cube[3][:,2] = temp[::-1]
    def U(self):
        # 윗면 시계 방향 회전
        self.cube[0] = np.rot90(self.cube[0], -1)
        
        # 인접 면 조각들 교환 (시계방향: L → F → R → B → L)
        temp = self.cube[4][0,:].copy()
        self.cube[4][0,:] = self.cube[2][0,:]
        self.cube[2][0,:] = self.cube[5][0,:]
        self.cube[5][0,:] = self.cube[3][0,:]
        self.cube[3][0,:] = temp
    def U_prime(self):
        # 윗면 반시계 방향 회전
        self.cube[0] = np.rot90(self.cube[0], 1)
        
        # 인접 면 조각들 교환 (반시계방향: L → B → R → F → L)
        temp = self.cube[4][0,:].copy()
        self.cube[4][0,:] = self.cube[3][0,:]
        self.cube[3][0,:] = self.cube[5][0,:]
        self.cube[5][0,:] = self.cube[2][0,:]
        self.cube[2][0,:] = temp
    def U_double(self):
        # 윗면 180도 회전
        self.cube[0] = np.rot90(self.cube[0], 2)
        
        # 인접 면 조각들 교환
        temp = self.cube[4][0,:].copy()
        self.cube[4][0,:] = self.cube[5][0,:]
        self.cube[5][0,:] = temp
        temp = self.cube[2][0,:].copy()
        self.cube[2][0,:] = self.cube[3][0,:]
        self.cube[3][0,:] = temp
    def F(self):
        # 앞면 시계 방향 회전
        self.cube[2] = np.rot90(self.cube[2], -1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][2,:].copy()
        self.cube[0][2,:] = self.cube[4][:,2][::-1]
        self.cube[4][:,2] = self.cube[1][0,:]
        self.cube[1][0,:] = self.cube[5][:,0][::-1]
        self.cube[5][:,0] = temp   
    def F_prime(self):
        # 앞면 반시계 방향 회전
        self.cube[2] = np.rot90(self.cube[2], 1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][2,:].copy()
        self.cube[0][2,:] = self.cube[5][:,0]
        self.cube[5][:,0] = self.cube[1][0,:][::-1]
        self.cube[1][0,:] = self.cube[4][:,2]
        self.cube[4][:,2] = temp[::-1]
    def F_double(self):
        # 앞면 180도 회전
        self.cube[2] = np.rot90(self.cube[2], 2)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][2,:].copy()
        self.cube[0][2,:] = self.cube[1][0,:]
        self.cube[1][0,:] = temp
        temp = self.cube[4][:,2].copy()
        self.cube[4][:,2] = self.cube[5][:,0][::-1]
        self.cube[5][:,0] = temp[::-1]
    def D(self):
        # 아랫면 시계 방향 회전 (아래에서 보면 시계방향)
        self.cube[1] = np.rot90(self.cube[1], -1)
        
        # 인접 면 조각들 교환 (아래에서 보면 시계방향: L → B → R → F → L)
        temp = self.cube[4][2,:].copy()
        self.cube[4][2,:] = self.cube[3][2,:]
        self.cube[3][2,:] = self.cube[5][2,:]
        self.cube[5][2,:] = self.cube[2][2,:]
        self.cube[2][2,:] = temp
    def D_prime(self):
        # 아랫면 반시계 방향 회전 (아래에서 보면 반시계방향)
        self.cube[1] = np.rot90(self.cube[1], 1)
        
        # 인접 면 조각들 교환 (아래에서 보면 반시계방향: L → F → R → B → L)
        temp = self.cube[4][2,:].copy()
        self.cube[4][2,:] = self.cube[2][2,:]
        self.cube[2][2,:] = self.cube[5][2,:]
        self.cube[5][2,:] = self.cube[3][2,:]
        self.cube[3][2,:] = temp
    def D_double(self):
        # 아랫면 180도 회전
        self.cube[1] = np.rot90(self.cube[1], 2)
        
        # 인접 면 조각들 교환
        temp = self.cube[4][2,:].copy()
        self.cube[4][2,:] = self.cube[5][2,:]
        self.cube[5][2,:] = temp
        temp = self.cube[2][2,:].copy()
        self.cube[2][2,:] = self.cube[3][2,:]
        self.cube[3][2,:] = temp
    def B(self):
        # 뒷면 시계 방향 회전
        self.cube[3] = np.rot90(self.cube[3], -1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][0,:].copy()
        self.cube[0][0,:] = self.cube[5][:,2]
        self.cube[5][:,2] = self.cube[1][2,:][::-1]
        self.cube[1][2,:] = self.cube[4][:,0]
        self.cube[4][:,0] = temp[::-1]
    def B_prime(self):
        # 뒷면 반시계 방향 회전
        self.cube[3] = np.rot90(self.cube[3], 1)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][0,:].copy()
        self.cube[0][0,:] = self.cube[4][:,0][::-1]
        self.cube[4][:,0] = self.cube[1][2,:]
        self.cube[1][2,:] = self.cube[5][:,2][::-1]
        self.cube[5][:,2] = temp
    def B_double(self):
        # 뒷면 180도 회전
        self.cube[3] = np.rot90(self.cube[3], 2)
        
        # 인접 면 조각들 교환
        temp = self.cube[0][0,:].copy()
        self.cube[0][0,:] = self.cube[1][2,:][::-1]
        self.cube[1][2,:] = temp
        temp = self.cube[5][:,2].copy()
        self.cube[5][:,2] = self.cube[4][:,0][::-1]
        self.cube[4][:,0] = temp[::-1]
    def x(self):
        # x 무브: R과 같은 방향으로 큐브 전체 회전
        
        # 1. 좌우 면 회전
        self.cube[5] = np.rot90(self.cube[5], -1)  # R면 시계방향
        self.cube[4] = np.rot90(self.cube[4], 1)   # L면 반시계방향
        
        # 2. 상하앞뒤 면 순환 (센터 포함 전체)
        temp = self.cube[0].copy()  # U 전체 저장
        self.cube[0] = self.cube[2]  # F → U
        self.cube[2] = self.cube[1]  # D → F
        self.cube[1] = self.cube[3]  # B → D
        self.cube[3] = temp          # U → B
    def x_prime(self):
        # x' 무브: R과 반대 방향으로 큐브 전체 회전
        
        # 1. 좌우 면 회전
        self.cube[5] = np.rot90(self.cube[5], 1)   # R면 반시계방향
        self.cube[4] = np.rot90(self.cube[4], -1)  # L면 시계방향
        
        # 2. 상하앞뒤 면 순환 (센터 포함 전체)
        temp = self.cube[0].copy()  # U 전체 저장
        self.cube[0] = self.cube[3]  # B → U
        self.cube[3] = self.cube[1]  # D → B
        self.cube[1] = self.cube[2]  # F → D
        self.cube[2] = temp          # U → F
    def x_double(self):
        # x2 무브: 큐브 전체 180도 회전
        
        # 1. 좌우 면 180도 회전
        self.cube[5] = np.rot90(self.cube[5], 2)  # R면 180도
        self.cube[4] = np.rot90(self.cube[4], 2)  # L면 180도
        
        # 2. 상하앞뒤 면 순환 (센터 포함 전체)
        temp = self.cube[0].copy()  # U 전체 저장
        self.cube[0] = self.cube[1]  # D → U
        self.cube[1] = temp          # U → D
        temp = self.cube[2].copy()  # F 전체 저장
        self.cube[2] = self.cube[3]  # B → F
        self.cube[3] = temp          # F → B
    def y(self):
        # y 무브: U와 같은 방향으로 큐브 전체 회전
        
        # 1. 상하 면 회전
        self.cube[0] = np.rot90(self.cube[0], -1)  # U면 시계방향
        self.cube[1] = np.rot90(self.cube[1], 1)   # D면 반시계방향
        
        # 2. 좌우앞뒤 면 순환 (센터 포함 전체)
        temp = self.cube[4].copy()  # L 전체 저장
        self.cube[4] = self.cube[2]  # F → L
        self.cube[2] = self.cube[5]  # R → F
        self.cube[5] = self.cube[3]  # B → R
        self.cube[3] = temp          # L → B
    def y_prime(self):
        # y' 무브: U와 반대 방향으로 큐브 전체 회전
        
        # 1. 상하 면 회전
        self.cube[0] = np.rot90(self.cube[0], 1)   # U면 반시계방향
        self.cube[1] = np.rot90(self.cube[1], -1)  # D면 시계방향
        
        # 2. 좌우앞뒤 면 순환 (센터 포함 전체)
        temp = self.cube[4].copy()  # L 전체 저장
        self.cube[4] = self.cube[3]  # B → L
        self.cube[3] = self.cube[5]  # R → B
        self.cube[5] = self.cube[2]  # F → R
        self.cube[2] = temp          # L → F
    def y_double(self):
        # y2 무브: 큐브 전체 180도 회전
        
        # 1. 상하 면 180도 회전
        self.cube[0] = np.rot90(self.cube[0], 2)  # U면 180도
        self.cube[1] = np.rot90(self.cube[1], 2)  # D면 180도
        
        # 2. 좌우앞뒤 면 순환 (센터 포함 전체)
        temp = self.cube[4].copy()  # L 전체 저장
        self.cube[4] = self.cube[5]  # R → L
        self.cube[5] = temp          # L → R
        temp = self.cube[2].copy()  # F 전체 저장
        self.cube[2] = self.cube[3]  # B → F
        self.cube[3] = temp          # F → B
    def z(self):
        # z 무브: F와 같은 방향으로 큐브 전체 회전
        
        # 1. 앞뒤 면 회전
        self.cube[2] = np.rot90(self.cube[2], -1)  # F면 시계방향
        self.cube[3] = np.rot90(self.cube[3], 1)   # B면 반시계방향
        
        # 2. 상하좌우 면 순환 (센터 포함 전체)
        temp = self.cube[0].copy()  # U 전체 저장
        self.cube[0] = self.cube[4][:,::-1].T  # L → U
        self.cube[4] = self.cube[1][:,::-1].T  # D → L
        self.cube[1] = self.cube[5][:,::-1].T  # R → D
        self.cube[5] = temp.T                  # U → R
    def z_prime(self):
        # z' 무브: F와 반대 방향으로 큐브 전체 회전
        
        # 1. 앞뒤 면 회전
        self.cube[2] = np.rot90(self.cube[2], 1)   # F면 반시계방향
        self.cube[3] = np.rot90(self.cube[3], -1)  # B면 시계방향
        
        # 2. 상하좌우 면 순환 (센터 포함 전체)
        temp = self.cube[0].copy()  # U 전체 저장
        self.cube[0] = self.cube[5].T                  # R → U
        self.cube[5] = self.cube[1][:,::-1].T          # D → R
        self.cube[1] = self.cube[4][:,::-1].T          # L → D
        self.cube[4] = temp[:,::-1].T                   # U → L
    def z_double(self):
        # z2 무브: 큐브 전체 180도 회전
        
        # 1. 앞뒤 면 180도 회전
        self.cube[2] = np.rot90(self.cube[2], 2)  # F면 180도
        self.cube[3] = np.rot90(self.cube[3], 2)  # B면 180도
        
        # 2. 상하좌우 면 순환 (센터 포함 전체)
        temp = self.cube[0].copy()  # U 전체 저장
        self.cube[0] = self.cube[1][:,::-1].T  # D → U
        self.cube[1] = temp[:,::-1].T          # U → D
        temp = self.cube[4].copy()  # L 전체 저장
        self.cube[4] = self.cube[5][:,::-1].T  # R → L
        self.cube[5] = temp[:,::-1].T          # L → R
    
    # ==================== 슬라이스 무브 (M, E, S) ====================
    def M(self):
        """M: 중간 세로 슬라이스 (L과 같은 방향으로 회전)"""
        # L과 R 사이의 중간 열만 회전 (인덱스 1)
        temp = self.cube[0][:,1].copy()
        self.cube[0][:,1] = self.cube[3][:,1][::-1]
        self.cube[3][:,1] = self.cube[1][:,1][::-1]
        self.cube[1][:,1] = self.cube[2][:,1]
        self.cube[2][:,1] = temp
    
    def M_prime(self):
        """M': 중간 세로 슬라이스 반시계 방향"""
        temp = self.cube[0][:,1].copy()
        self.cube[0][:,1] = self.cube[2][:,1]
        self.cube[2][:,1] = self.cube[1][:,1]
        self.cube[1][:,1] = self.cube[3][:,1][::-1]
        self.cube[3][:,1] = temp[::-1]
    
    def M_double(self):
        """M2: 중간 세로 슬라이스 180도"""
        temp = self.cube[0][:,1].copy()
        self.cube[0][:,1] = self.cube[1][:,1]
        self.cube[1][:,1] = temp
        temp = self.cube[2][:,1].copy()
        self.cube[2][:,1] = self.cube[3][:,1][::-1]
        self.cube[3][:,1] = temp[::-1]
    
    def E(self):
        """E: 중간 가로 슬라이스 (D와 같은 방향으로 회전)"""
        # U와 D 사이의 중간 행만 회전 (인덱스 1)
        temp = self.cube[4][1,:].copy()
        self.cube[4][1,:] = self.cube[3][1,:]
        self.cube[3][1,:] = self.cube[5][1,:]
        self.cube[5][1,:] = self.cube[2][1,:]
        self.cube[2][1,:] = temp
    
    def E_prime(self):
        """E': 중간 가로 슬라이스 반시계 방향"""
        temp = self.cube[4][1,:].copy()
        self.cube[4][1,:] = self.cube[2][1,:]
        self.cube[2][1,:] = self.cube[5][1,:]
        self.cube[5][1,:] = self.cube[3][1,:]
        self.cube[3][1,:] = temp
    
    def E_double(self):
        """E2: 중간 가로 슬라이스 180도"""
        temp = self.cube[4][1,:].copy()
        self.cube[4][1,:] = self.cube[5][1,:]
        self.cube[5][1,:] = temp
        temp = self.cube[2][1,:].copy()
        self.cube[2][1,:] = self.cube[3][1,:]
        self.cube[3][1,:] = temp
    
    def S(self):
        """S: 중간 전후 슬라이스 (F와 같은 방향으로 회전)"""
        # F와 B 사이의 중간 슬라이스만 회전 (인덱스 1)
        temp = self.cube[0][1,:].copy()
        self.cube[0][1,:] = self.cube[4][:,1][::-1]
        self.cube[4][:,1] = self.cube[1][1,:]
        self.cube[1][1,:] = self.cube[5][:,1][::-1]
        self.cube[5][:,1] = temp
    
    def S_prime(self):
        """S': 중간 전후 슬라이스 반시계 방향"""
        temp = self.cube[0][1,:].copy()
        self.cube[0][1,:] = self.cube[5][:,1]
        self.cube[5][:,1] = self.cube[1][1,:][::-1]
        self.cube[1][1,:] = self.cube[4][:,1]
        self.cube[4][:,1] = temp[::-1]
    
    def S_double(self):
        """S2: 중간 전후 슬라이스 180도"""
        temp = self.cube[0][1,:].copy()
        self.cube[0][1,:] = self.cube[1][1,:]
        self.cube[1][1,:] = temp
        temp = self.cube[4][:,1].copy()
        self.cube[4][:,1] = self.cube[5][:,1][::-1]
        self.cube[5][:,1] = temp[::-1]
    
    # ==================== 와이드 무브 (기본 무브 + 슬라이스 조합) ====================
    def r(self):
        """r = R + M'"""
        self.R()
        self.M_prime()
    
    def r_prime(self):
        """r' = R' + M"""
        self.R_prime()
        self.M()
    
    def r_double(self):
        """r2 = R2 + M2"""
        self.R_double()
        self.M_double()
    
    def l(self):
        """l = L + M"""
        self.L()
        self.M()
    
    def l_prime(self):
        """l' = L' + M'"""
        self.L_prime()
        self.M_prime()
    
    def l_double(self):
        """l2 = L2 + M2"""
        self.L_double()
        self.M_double()
    
    def u(self):
        """u = U + E'"""
        self.U()
        self.E_prime()
    
    def u_prime(self):
        """u' = U' + E"""
        self.U_prime()
        self.E()
    
    def u_double(self):
        """u2 = U2 + E2"""
        self.U_double()
        self.E_double()
    
    def d(self):
        """d = D + E"""
        self.D()
        self.E()
    
    def d_prime(self):
        """d' = D' + E'"""
        self.D_prime()
        self.E_prime()
    
    def d_double(self):
        """d2 = D2 + E2"""
        self.D_double()
        self.E_double()
    
    def f(self):
        """f = F + S"""
        self.F()
        self.S()
    
    def f_prime(self):
        """f' = F' + S'"""
        self.F_prime()
        self.S_prime()
    
    def f_double(self):
        """f2 = F2 + S2"""
        self.F_double()
        self.S_double()
    
    def b(self):
        """b = B + S'"""
        self.B()
        self.S_prime()
    
    def b_prime(self):
        """b' = B' + S"""
        self.B_prime()
        self.S()
    
    def b_double(self):
        """b2 = B2 + S2"""
        self.B_double()
        self.S_double()
    
    def scramble(self, moves_count=20):
        Last_move = -1
        
        import random
        Move_s = ["R", "R'", "R2",
                   "L", "L'", "L2",
                   "U", "U'", "U2",
                   "D", "D'", "D2",
                   "F", "F'", "F2",
                   "B", "B'", "B2"]
        move_method = [self.R, self.R_prime, self.R_double,
                       self.L, self.L_prime, self.L_double,
                       self.U, self.U_prime, self.U_double,
                       self.D, self.D_prime, self.D_double,
                       self.F, self.F_prime, self.F_double,
                       self.B, self.B_prime, self.B_double]
        
        print(f"스크램블 시작 ({moves_count}개 무브):")
        for i in range(moves_count):
            # 무브 인덱스 랜덤 선택
            idx = random.randint(0, len(move_method) - 1)
            if Last_move // 3 == idx // 3:  # 같은 면 무브 방지
                while Last_move // 3 == idx // 3:
                    idx = random.randint(0, len(move_method) - 1)
            move_name = Move_s[idx]
            move_func = move_method[idx]
            
            # 무브 실행 및 출력
            print(f"  {i+1}. {move_name}", end=' ')
            move_func()
            Last_move = idx

            
    def reset(self):
        self.cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],
                              [[2,2,2],[2,2,2],[2,2,2]],
                              [[3,3,3],[3,3,3],[3,3,3]],
                              [[4,4,4],[4,4,4],[4,4,4]],
                              [[5,5,5],[5,5,5],[5,5,5]],
                              [[6,6,6],[6,6,6],[6,6,6]]])
    

def test_moves(r):
    moves = [
        ('R', r.R),
        ("R'", r.R_prime),
        ('R2', r.R_double),
        ('L', r.L),
        ("L'", r.L_prime),
        ('L2', r.L_double),
        ('U', r.U),
        ("U'", r.U_prime),
        ('U2', r.U_double),
        ('D', r.D),
        ("D'", r.D_prime),
        ('D2', r.D_double),
        ('F', r.F),
        ("F'", r.F_prime),
        ('F2', r.F_double),
        ('B', r.B),
        ("B'", r.B_prime),
        ('B2', r.B_double)
    ]
    
    for move_name, move_func in moves:
        move_func()
        print("\n" + "="*60)
        print(f"After move: {move_name}")
        print("="*60)
        draw_cube_flat(color, r.cube)
        r.reset()


# 테스트 실행
r = Moves(Rubik_cube)

print("\n" + "="*60)
print("초기 상태")
print("="*60)
draw_cube_flat(color, Rubik_cube)
r.scramble()
draw_cube_flat(color, Rubik_cube)



# 모든 무브 테스트
