from typing import Optional, Sequence, NamedTuple, Any

import numpy as np

from .abc import Chessboard


class BitChessboard(Chessboard):

    @classmethod
    def from_board_state(
            cls,
            board: Sequence[Sequence[int]],
            castling_stats: Sequence[Sequence[int]],
            turn: int,
            fifty_move_count: int,
            enpassant_file: int,
            ply_count: int,
    ) -> Chessboard:

        bitboards = np.empty(shape=(13, ))