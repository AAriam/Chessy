from typing import Optional, Sequence, NamedTuple, Any

import numpy as np

from .abc import Judge


class BitChessboard(Judge):
    @classmethod
    def load_state(
        cls,
        board: Sequence[Sequence[int]],
        castling_stats: Sequence[Sequence[int]],
        turn: int,
        fifty_move_count: int,
        enpassant_file: int,
        ply_count: int,
    ) -> Judge:

        bitboards = np.empty(shape=(13,))
