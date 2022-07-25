import numpy as np


def new_board() -> np.ndarray:
    """
    Create a chessboard in starting position.
    """
    board = np.zeros((8, 8), dtype=np.int8)  # Initialize an all-zero 8x8 array
    board[1, :] = 1  # Set white pawns on row 2
    board[-2, :] = -1  # Set black pawns on row 7
    board[0, :] = [4, 2, 3, 5, 6, 3, 2, 4]  # Set white's main pieces on row 1
    board[-1, :] = -board[0]  # Set black's main pieces on row 8
    return board
