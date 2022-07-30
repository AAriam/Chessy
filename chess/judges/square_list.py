from typing import Optional, Sequence, NamedTuple, Any

import numpy as np

from .abc import Chessboard


class ArrayChessboard(Chessboard):

    def __init__(
            self,
            board: Sequence[Sequence[int]],
            castling_stats: Sequence[Sequence[int]],
            turn: int,
            fifty_move_count: int,
            enpassant_file: int,
            ply_count: int,
    ):
        super().__init__()
        self._board: np.ndarray = np.array(board, dtype=np.int8)
        self._can_castle: np.ndarray = np.array(castling_stats, dtype=np.int8)
        self._turn: np.int8 = np.int8(turn)
        self._fifty_move_count: np.int8 = np.int8(fifty_move_count)
        self._enpassant_file: np.int8 = np.int8(enpassant_file)
        self._ply_count: np.int16 = np.int16(ply_count)


    @property
    def board(self) -> np.ndarray:
        """
        Each element thus corresponds to a square, e.g. `board[0, 0]` corresponds
        to square 'a1', `board[0, 7]` to 'h1', and `board[7, 7]` to 'h8'.
        The elements are of type `numpy.byte`, and contain information about that square:
        0: empty, 1: pawn, 2: knight, 3: bishop, 4: rook, 5: queen, 6: king
        White pieces are denoted with positive integers, while black pieces have the
        same magnitude but with a negative sign.
        """
        return self._board

    @property
    def turn(self) -> int:
        """
        Whose turn it is to move, described as an integer:
        +1 for white, and -1 for black
        """
        return self._turn