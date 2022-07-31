from .boards.abc import BoardState

import numpy as np


def boardstate_initial_standard() -> BoardState:
    """
    Instantiate a new Chessboard in the starting position of a standard game.

    Returns
    -------
    BoardState
    """
    # Set up board
    board = np.zeros(shape=(8, 8), dtype=np.int8)  # Initialize an all-zero 8x8 array
    board[(1, -2),] = [1], [-1]  # Set white and black pawns on rows 2 and 7
    board[0, :] = [4, 2, 3, 5, 6, 3, 2, 4]  # Set white's main pieces on row 1
    board[-1, :] = -board[0]  # Set black's main pieces on row 8
    # Set instance attributes describing the game state to their initial values
    return BoardState(
        board=board,
        castling_rights=np.ones(shape=(2, 2), dtype=np.int8),
        player=np.int8(1),
        enpassant_file=np.int8(-1),
        fifty_move_count=np.int8(0),
        ply_count=np.int16(0),
    )


def main():
    pass


if __name__ == "__main__":
    main()
