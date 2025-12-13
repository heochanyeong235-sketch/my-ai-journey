import numpy as np
from visualize import Rubik_cube, Moves, draw_cube_flat, color
from ai_env import Rubiks, corner_orient, corner_permu, edge_orient, edge_permu, corner_index, edge_index

# ======================
# 0~5 숫자로 각 면 정의
# ======================
U, R, F, D, L, B = 1,6,3,2,5,4 

# ======================
# 코너 / 엣지 색 조합 (0~5 숫자 기준)
# ======================
corner_color_table = {
    0: [U, F, R],  # UFR
    1: [U, R, B],  # URB
    2: [U, B, L],  # UBL
    3: [U, L, F],  # ULF
    4: [D, F, R],  # DFR
    5: [D, R, B],  # DRB
    6: [D, B, L],  # DBL
    7: [D, L, F],  # DLF
}

edge_color_table = {
    0:  [U, F],  # UF
    1:  [U, R],  # UR
    2:  [U, B],  # UB
    3:  [U, L],  # UL
    4:  [F, R],  # FR
    5:  [R, B],  # RB
    6:  [B, L],  # BL
    7:  [L, F],  # LF
    8:  [D, F],  # DF
    9:  [D, R],  # DR
    10: [D, B],  # DB
    11: [D, L],  # DL
}

# ======================
# 코너 / 엣지 스티커 위치
# ======================
corner_indices = {
    'UFR': [(0,2,2), (2,0,2), (5,0,0)],
    'URB': [(0,0,2), (5,0,2), (3,0,0)],
    'UBL': [(0,0,0), (3,0,2), (4,0,0)],
    'ULF': [(0,2,0), (4,0,2), (2,0,0)],
    'DFR': [(1,0,2), (2,2,2), (5,2,0)],
    'DRB': [(1,2,2), (5,2,2), (3,2,0)],
    'DBL': [(1,2,0), (3,2,2), (4,2,0)],
    'DLF': [(1,0,0), (4,2,2), (2,2,0)],
}

edge_indices = {
    'UF': [(0,2,1), (2,0,1)],
    'UR': [(0,1,2), (5,0,1)],
    'UB': [(0,0,1), (3,0,1)],
    'UL': [(0,1,0), (4,0,1)],
    'FR': [(2,1,2), (5,1,0)],
    'RB': [(5,1,2), (3,1,0)],
    'BL': [(3,1,2), (4,1,0)],
    'LF': [(4,1,2), (2,1,0)],
    'DF': [(1,0,1), (2,2,1)],
    'DR': [(1,1,2), (5,2,1)],
    'DB': [(1,2,1), (3,2,1)],
    'DL': [(1,1,0), (4,2,1)],
}

# ======================
# 스티커 읽기 함수
# ======================
def get_corner_stickers(cube, name):
    pos = corner_indices[name]
    return [int(cube[f][r][c]) for f, r, c in pos]

def get_edge_stickers(cube, name):
    pos = edge_indices[name]
    return [int(cube[f][r][c]) for f, r, c in pos]

# ======================
# 전체 코너/엣지 상태 가져오기
# ======================
def get_all_corners(cube):
    corners = [None]*8
    for name, idx in corner_index.items():
        corners[idx] = get_corner_stickers(cube, name)
    return corners

def get_all_edges(cube):
    edges = [None]*12
    for name, idx in edge_index.items():
        edges[idx] = get_edge_stickers(cube, name)
    return edges

# ======================
# 퍼뮤테이션 / 오리엔테이션 식별
# ======================
def identify_corner_piece(sticker):
    s = sorted(sticker)
    for piece_id, colors in corner_color_table.items():
        if s == sorted(colors):
            return piece_id
    raise ValueError("Invalid corner")

def identify_corner_orientation(sticker):
    piece_id = identify_corner_piece(sticker)
    colors = corner_color_table[piece_id]
    if sticker[0] == colors[0]:
        return 0
    elif sticker[1] == colors[0]:
        return 1
    else:
        return 2

# ======================
# 예제: Rubik_cube 상태 -> AI 배열 변환
# ======================
corners = get_all_corners(Rubik_cube)
edges = get_all_edges(Rubik_cube)

for pos, sticker in enumerate(corners):
    piece_id = identify_corner_piece(sticker)
    orient = identify_corner_orientation(sticker)
    corner_permu[pos] = piece_id
    corner_orient[pos] = orient

print("Corner Permutation:", corner_permu)
print("Corner Orientation:", corner_orient)

# 엣지도 동일하게 처리 가능 (필요 시 identify_edge_piece/identify_edge_orientation 구현)
