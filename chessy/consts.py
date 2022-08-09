

import numpy as np


# Ranks
RANK_1 = np.int8(0)
RANK_2 = np.int8(1)
RANK_3 = np.int8(2)
RANK_4 = np.int8(3)
RANK_5 = np.int8(4)
RANK_6 = np.int8(5)
RANK_7 = np.int8(6)
RANK_8 = np.int8(7)

# Files
FILE_A = np.int8(0)
FILE_B = np.int8(1)
FILE_C = np.int8(2)
FILE_D = np.int8(3)
FILE_E = np.int8(4)
FILE_F = np.int8(5)
FILE_G = np.int8(6)
FILE_H = np.int8(7)

# Squares
# generated with:
# for f, file in enumerate("abcdefgh"):
#     for r, rank in enumerate(range(1,9)):
#         print(f"{file.upper()}{rank} = (np.int8({r}), np.int8({f}))")
A1 = (np.int8(0), np.int8(0))
A2 = (np.int8(1), np.int8(0))
A3 = (np.int8(2), np.int8(0))
A4 = (np.int8(3), np.int8(0))
A5 = (np.int8(4), np.int8(0))
A6 = (np.int8(5), np.int8(0))
A7 = (np.int8(6), np.int8(0))
A8 = (np.int8(7), np.int8(0))
B1 = (np.int8(0), np.int8(1))
B2 = (np.int8(1), np.int8(1))
B3 = (np.int8(2), np.int8(1))
B4 = (np.int8(3), np.int8(1))
B5 = (np.int8(4), np.int8(1))
B6 = (np.int8(5), np.int8(1))
B7 = (np.int8(6), np.int8(1))
B8 = (np.int8(7), np.int8(1))
C1 = (np.int8(0), np.int8(2))
C2 = (np.int8(1), np.int8(2))
C3 = (np.int8(2), np.int8(2))
C4 = (np.int8(3), np.int8(2))
C5 = (np.int8(4), np.int8(2))
C6 = (np.int8(5), np.int8(2))
C7 = (np.int8(6), np.int8(2))
C8 = (np.int8(7), np.int8(2))
D1 = (np.int8(0), np.int8(3))
D2 = (np.int8(1), np.int8(3))
D3 = (np.int8(2), np.int8(3))
D4 = (np.int8(3), np.int8(3))
D5 = (np.int8(4), np.int8(3))
D6 = (np.int8(5), np.int8(3))
D7 = (np.int8(6), np.int8(3))
D8 = (np.int8(7), np.int8(3))
E1 = (np.int8(0), np.int8(4))
E2 = (np.int8(1), np.int8(4))
E3 = (np.int8(2), np.int8(4))
E4 = (np.int8(3), np.int8(4))
E5 = (np.int8(4), np.int8(4))
E6 = (np.int8(5), np.int8(4))
E7 = (np.int8(6), np.int8(4))
E8 = (np.int8(7), np.int8(4))
F1 = (np.int8(0), np.int8(5))
F2 = (np.int8(1), np.int8(5))
F3 = (np.int8(2), np.int8(5))
F4 = (np.int8(3), np.int8(5))
F5 = (np.int8(4), np.int8(5))
F6 = (np.int8(5), np.int8(5))
F7 = (np.int8(6), np.int8(5))
F8 = (np.int8(7), np.int8(5))
G1 = (np.int8(0), np.int8(6))
G2 = (np.int8(1), np.int8(6))
G3 = (np.int8(2), np.int8(6))
G4 = (np.int8(3), np.int8(6))
G5 = (np.int8(4), np.int8(6))
G6 = (np.int8(5), np.int8(6))
G7 = (np.int8(6), np.int8(6))
G8 = (np.int8(7), np.int8(6))
H1 = (np.int8(0), np.int8(7))
H2 = (np.int8(1), np.int8(7))
H3 = (np.int8(2), np.int8(7))
H4 = (np.int8(3), np.int8(7))
H5 = (np.int8(4), np.int8(7))
H6 = (np.int8(5), np.int8(7))
H7 = (np.int8(6), np.int8(7))
H8 = (np.int8(7), np.int8(7))

# Pieces
NULL = np.int8(0)
PAWN = np.int8(1)
KNIGHT = np.int8(2)
BISHOP = np.int8(3)
ROOK = np.int8(4)
QUEEN = np.int8(5)
KING = np.int8(6)

# Players
WHITE = np.int8(1)
BLACK = np.int8(-1)

# Castling
QUEENSIDE = np.int8(-2)
KINGSIDE = np.int8(2)
CASTLING_RIGHTS_DEFAULT = {
    WHITE: {QUEENSIDE: True, KINGSIDE: True},
    BLACK: {QUEENSIDE: True, KINGSIDE: True}
}

# Directions
TOP = [np.int8(1), np.int8(0)]
TOP_RIGHT = [np.int8(1), np.int8(1)]
RIGHT = [np.int8(0), np.int8(1)]
BOTTOM_RIGHT = [np.int8(-1), np.int8(1)]
BOTTOM = [np.int8(-1), np.int8(0)]
BOTTOM_LEFT = [np.int8(-1), np.int8(-1)]
LEFT = [np.int8(0), np.int8(-1)]
TOP_LEFT = [np.int8(1), np.int8(-1)]
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
    BLACK: np.array([B8, C8, D8, F8, G8], dtype=np.int8),
}
# Squares that must not be under attack for castling to be allowed for each player.
# First two squares correspond to queenside castle, and the next two to kingside castle.
CASTLING_SS_CHECK = {
    WHITE: np.array([C1, D1, F1, G1], dtype=np.int8),
    BLACK: np.array([C8, D8, F8, G8], dtype=np.int8),
}

