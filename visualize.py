import numpy as np
color = {"w":1, "y":2, "g":3, "b":4, "o":5, "r":6}
Rubik_cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]],[[3,3,3],[3,3,3],[3,3,3]],[[4,4,4],[4,4,4],[4,4,4]],[[5,5,5],[5,5,5],[5,5,5]],[[6,6,6],[6,6,6],[6,6,6]]])
print(Rubik_cube.shape)
def draw_cube_flat(color, Rubik_cube):
    print("\n전개도:")
    print()
    for row in Rubik_cube[0]:
        print("      " + ' '.join([list(color.keys())[list(color.values()).index(val)] for val in row]))
    for i in range(3):
        line = ""
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[4][i]])
        line += " "
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[2][i]])
        line += " "
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[5][i]])
        line += " "
        line += ' '.join([list(color.keys())[list(color.values()).index(val)] for val in Rubik_cube[3][i]])
        print(line)
    print("      ", end="")
    for row in Rubik_cube[1]:
        print(' '.join([list(color.keys())[list(color.values()).index(val)] for val in row]))
        if row is not Rubik_cube[1][-1]:
            print("      ", end="")
    print("")
class Moves:
    def __init__(self, Rubik_cube):
        self.cube = Rubik_cube
    def R(self):
        self.cube[5] = np.rot90(self.cube[5], -1)
        temp = self.cube[0][:,2].copy()
        self.cube[0][:,2] = self.cube[2][:,2]
        self.cube[2][:,2] = self.cube[1][:,2]
        self.cube[1][:,2] = self.cube[3][:,0][::-1]
        self.cube[3][:,0] = temp[::-1]
    def R_prime(self):
        for i in range(3):
            self.R()
    def R_double(self):
        for i in range(2):
            self.R()
    def L(self):
        self.cube[4] = np.rot90(self.cube[4], -1)
        temp = self.cube[0][:,0].copy()
        self.cube[0][:,0] = self.cube[3][:,2][::-1]
        self.cube[3][:,2] = self.cube[1][:,0][::-1]
        self.cube[1][:,0] = self.cube[2][:,0]
        self.cube[2][:,0] = temp
    def L_prime(self):
        for i in range(3):
            self.L()
    def L_double(self):
        # 왼쪽 면 180도 회전
        for i in range(2):
            self.L()
    def U(self):
        self.cube[0] = np.rot90(self.cube[0], -1)
        temp = self.cube[4][0,:].copy()
        self.cube[4][0,:] = self.cube[2][0,:]
        self.cube[2][0,:] = self.cube[5][0,:]
        self.cube[5][0,:] = self.cube[3][0,:]
        self.cube[3][0,:] = temp
    def U_prime(self):
        for i in range(3):
            self.U()
    def U_double(self):
        for i in range(2):
            self.U()
    def F(self):
        self.cube[2] = np.rot90(self.cube[2], -1)
        temp = self.cube[0][2,:].copy()
        self.cube[0][2,:] = self.cube[4][:,2][::-1]
        self.cube[4][:,2] = self.cube[1][0,:]
        self.cube[1][0,:] = self.cube[5][:,0][::-1]
        self.cube[5][:,0] = temp   
    def F_prime(self):
        for i in range(3):
            self.F()
    def F_double(self):
        for i in range(2):
            self.F()
    def D(self):
        self.cube[1] = np.rot90(self.cube[1], -1)
        temp = self.cube[4][2,:].copy()
        self.cube[4][2,:] = self.cube[3][2,:]
        self.cube[3][2,:] = self.cube[5][2,:]
        self.cube[5][2,:] = self.cube[2][2,:]
        self.cube[2][2,:] = temp
    def D_prime(self):
        for i in range(3):
            self.D()
    def D_double(self):
        for i in range(2):
            self.D()
    def B(self):
        self.cube[3] = np.rot90(self.cube[3], -1)
        temp = self.cube[0][0,:].copy()
        self.cube[0][0,:] = self.cube[5][:,2]
        self.cube[5][:,2] = self.cube[1][2,:][::-1]
        self.cube[1][2,:] = self.cube[4][:,0]
        self.cube[4][:,0] = temp[::-1]
    def B_prime(self):
        for i in range(3):
            self.B()
    def B_double(self):
        for i in range(2):
            self.B()
    def x(self):
        self.cube[5] = np.rot90(self.cube[5], -1)
        self.cube[4] = np.rot90(self.cube[4], 1)
        temp = self.cube[0].copy()
        self.cube[0] = self.cube[2]
        self.cube[2] = self.cube[1]
        self.cube[1] = self.cube[3]
        self.cube[3] = temp
    def x_prime(self):
        for i in range(3):
            self.x()
    def x_double(self):
        for i in range(2):
            self.x()
    def y(self):
        self.cube[0] = np.rot90(self.cube[0], -1)
        self.cube[1] = np.rot90(self.cube[1], 1)
        temp = self.cube[4].copy()
        self.cube[4] = self.cube[2]
        self.cube[2] = self.cube[5]
        self.cube[5] = self.cube[3]
        self.cube[3] = temp
    def y_prime(self):
        for i in range(3):
            self.y()
    def y_double(self):
        for i in range(2):
            self.y()
    def z(self):
        self.cube[2] = np.rot90(self.cube[2], -1)
        self.cube[3] = np.rot90(self.cube[3], 1)
        temp = self.cube[0].copy()
        self.cube[0] = self.cube[4][:,::-1].T
        self.cube[4] = self.cube[1][:,::-1].T
        self.cube[1] = self.cube[5][:,::-1].T
        self.cube[5] = temp.T
    def z_prime(self):
        for i in range(3):
            self.z()
    def z_double(self):
        for i in range(2):
            self.z()
    # ==================== 슬라이스 무브 (M, E, S) ====================
    def M(self):
        temp = self.cube[0][:,1].copy()
        self.cube[0][:,1] = self.cube[3][:,1][::-1]
        self.cube[3][:,1] = self.cube[1][:,1][::-1]
        self.cube[1][:,1] = self.cube[2][:,1]
        self.cube[2][:,1] = temp   
    def M_prime(self):
        for i in range(3):
            self.M()   
    def M_double(self):
        for i in range(2):
            self.M()
    def E(self):
        temp = self.cube[4][1,:].copy()
        self.cube[4][1,:] = self.cube[3][1,:]
        self.cube[3][1,:] = self.cube[5][1,:]
        self.cube[5][1,:] = self.cube[2][1,:]
        self.cube[2][1,:] = temp
    def E_prime(self):
        for i in range(3):
            self.E()
    def E_double(self):
        for i in range(2):
            self.E()
    def S(self):
        temp = self.cube[0][1,:].copy()
        self.cube[0][1,:] = self.cube[4][:,1][::-1]
        self.cube[4][:,1] = self.cube[1][1,:]
        self.cube[1][1,:] = self.cube[5][:,1][::-1]
        self.cube[5][:,1] = temp 
    def S_prime(self):
        for i in range(3):
            self.S()
    def S_double(self):
        for i in range(2):
            self.S()
    def r(self):
        self.R()
        self.M_prime()
    def r_prime(self):
        self.R_prime()
        self.M()
    def r_double(self):
        self.R_double()
        self.M_double()
    def l(self):
        self.L()
        self.M()
    def l_prime(self):
        self.L_prime()
        self.M_prime()
    def l_double(self):
        self.L_double()
        self.M_double()
    def u(self):
        self.U()
        self.E_prime()
    def u_prime(self):
        self.U_prime()
        self.E()
    def u_double(self):
        self.U_double()
        self.E_double()
    def d(self):
        self.D()
        self.E()
    def d_prime(self):
        self.D_prime()
        self.E_prime()
    def d_double(self):
        self.D_double()
        self.E_double()
    def f(self):
        self.F()
        self.S()
    def f_prime(self):
        self.F_prime()
        self.S_prime()
    def f_double(self):
        self.F_double()
        self.S_double()
    def b(self):
        self.B()
        self.S_prime()
    def b_prime(self):
        self.B_prime()
        self.S()
    def b_double(self):
        self.B_double()
        self.S_double() 
    def scramble(self, moves_count=20):
        Last_move = -1
        import random
        Move_s = ["R", "R'", "R2","L", "L'", "L2","U", "U'", "U2","D", "D'", "D2","F", "F'", "F2","B", "B'", "B2"]
        move_method = [self.R, self.R_prime, self.R_double,self.L, self.L_prime, self.L_double,self.U, self.U_prime, self.U_double,self.D, self.D_prime, self.D_double,self.F, self.F_prime, self.F_double,self.B, self.B_prime, self.B_double]
        for i in range(moves_count):
            idx = random.randint(0, len(move_method) - 1)
            if Last_move // 3 == idx // 3:
                while Last_move // 3 == idx // 3:
                    idx = random.randint(0, len(move_method) - 1)
            move_name = Move_s[idx]
            print(move_name, end=' ')
            move_method[idx]()
            Last_move = idx         
    def reset(self):
        self.cube = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]],[[3,3,3],[3,3,3],[3,3,3]],[[4,4,4],[4,4,4],[4,4,4]],[[5,5,5],[5,5,5],[5,5,5]],[[6,6,6],[6,6,6],[6,6,6]]])  
def test_moves(r):
    moves = [('R', r.R),("R'", r.R_prime),('R2', r.R_double),('L', r.L),("L'", r.L_prime),('L2', r.L_double),('U', r.U),("U'", r.U_prime),('U2', r.U_double),('D', r.D),("D'", r.D_prime),('D2', r.D_double),('F', r.F),("F'", r.F_prime),('F2', r.F_double),('B', r.B),("B'", r.B_prime),('B2', r.B_double)]
    for move_name, move_func in moves:
        move_func()
        print("\n" + "="*60)
        print(f"After move: {move_name}")
        print("="*60)
        draw_cube_flat(color, r.cube)
        r.reset()
r = Moves(Rubik_cube)
r.scramble()
draw_cube_flat(color, Rubik_cube)
