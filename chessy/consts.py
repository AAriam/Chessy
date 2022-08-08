

import numpy as np


# Ranks
RANK_1 = 0
RANK_2 = 1
RANK_3 = 2
RANK_4 = 3
RANK_5 = 4
RANK_6 = 5
RANK_7 = 6
RANK_8 = 7

# Files
FILE_A = 0
FILE_B = 1
FILE_C = 2
FILE_D = 3
FILE_E = 4
FILE_F = 5
FILE_G = 6
FILE_H = 7

# Squares
A1 = (0, 0)
A2 = (1, 0)
A3 = (2, 0)
A4 = (3, 0)
A5 = (4, 0)
A6 = (5, 0)
A7 = (6, 0)
A8 = (7, 0)
B1 = (0, 1)
B2 = (1, 1)
B3 = (2, 1)
B4 = (3, 1)
B5 = (4, 1)
B6 = (5, 1)
B7 = (6, 1)
B8 = (7, 1)
C1 = (0, 2)
C2 = (1, 2)
C3 = (2, 2)
C4 = (3, 2)
C5 = (4, 2)
C6 = (5, 2)
C7 = (6, 2)
C8 = (7, 2)
D1 = (0, 3)
D2 = (1, 3)
D3 = (2, 3)
D4 = (3, 3)
D5 = (4, 3)
D6 = (5, 3)
D7 = (6, 3)
D8 = (7, 3)
E1 = (0, 4)
E2 = (1, 4)
E3 = (2, 4)
E4 = (3, 4)
E5 = (4, 4)
E6 = (5, 4)
E7 = (6, 4)
E8 = (7, 4)
F1 = (0, 5)
F2 = (1, 5)
F3 = (2, 5)
F4 = (3, 5)
F5 = (4, 5)
F6 = (5, 5)
F7 = (6, 5)
F8 = (7, 5)
G1 = (0, 6)
G2 = (1, 6)
G3 = (2, 6)
G4 = (3, 6)
G5 = (4, 6)
G6 = (5, 6)
G7 = (6, 6)
G8 = (7, 6)
H1 = (0, 7)
H2 = (1, 7)
H3 = (2, 7)
H4 = (3, 7)
H5 = (4, 7)
H6 = (5, 7)
H7 = (6, 7)
H8 = (7, 7)

# Pieces
NULL = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

# Players
WHITE = 1
BLACK = -1

# Castling
QUEENSIDE = -2
KINGSIDE = 2

# Directions
TOP = [1, 0]
TOP_RIGHT = [1, 1]
RIGHT = [0, 1]
BOTTOM_RIGHT = [-1, 1]
BOTTOM = [-1, 0]
BOTTOM_LEFT = [-1, -1]
LEFT = [0, -1]
TOP_LEFT = [1, -1]
DIRECTIONS_ORTHO = np.array([TOP, RIGHT, BOTTOM, LEFT], dtype=np.int8)
DIRECTIONS_DIAG = np.array([TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, TOP_LEFT], dtype=np.int8)
DIRECTIONS = np.array(
        [TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT, LEFT, TOP_LEFT], dtype=np.int8
    )
MOVE_DIRECTIONS = {
    PAWN: np.array([TOP, [2, 0], TOP_RIGHT, TOP_LEFT], dtype=np.int8),
    KNIGHT: np.array(
        [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]], dtype=np.int8
    ),
    BISHOP: DIRECTIONS_DIAG,
    ROOK: DIRECTIONS_ORTHO,
    QUEEN: np.concatenate([DIRECTIONS_ORTHO, DIRECTIONS_DIAG]),
    KING: np.concatenate([DIRECTIONS, np.array([[0, -2], [0, 2]], dtype=np.int8)]),
}

# Positions
RANK_PAWN = {WHITE: RANK_2, BLACK: RANK_7}
RANK_ENPASSANT_END = {WHITE: RANK_6, BLACK: RANK_3}
RANK_PROMOTION = {WHITE: RANK_8, BLACK: RANK_1}
ROOK_S_INIT = {
    WHITE: {QUEENSIDE: A1, KINGSIDE: H1},
    BLACK: {QUEENSIDE: A8, KINGSIDE: H8}
}
ROOK_S_END = {
    WHITE: {QUEENSIDE: D1, KINGSIDE: F1},
    BLACK: {QUEENSIDE: D8, KINGSIDE: F8}
}
# Squares that must be empty for each player for castling to be allowed. First three squares
# correspond to queenside castle, and the next two correspond to kingside castle.
CASTLING_SS_EMPTY = {
    WHITE: np.array([B1, C1, D1, F1, G1], dtype=np.int8),
    BLACK: np.array([B8, C8, D8, F8, G1], dtype=np.int8),
}
# Squares that must not be under attack for castling to be allowed for each player.
# First two squares correspond to queenside castle, and the next two to kingside castle.
CASTLING_SS_CHECK = {
    WHITE: np.array([C1, D1, F1, G1], dtype=np.int8),
    BLACK: np.array([C8, D8, F8, G8], dtype=np.int8),
}

