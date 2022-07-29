from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod

from .. import algebraic_notation


class Player(ABC):

    def __init__(self, name, value):
        self._name = name
        self._value = value
        return

    def __call__(self):
        return self._value

    @property
    def name(self):
        return self._name


class Piece(ABC):
    ...


class Chessboard(ABC):

    def __init__(
            self,
            board: list[list[int]],
            castling_stats: list[list[int]],
            turn: int = 1,
            fifty_move_count: int = 0,
            enpassant_file: int = -1,
            ply_count: int = 0,
    ):
        """
        Create a new board from given data.

        Parameters
        ----------
        board : list[list[int]]
            A 2-dimensional list (list of lists) of integers, representing a specific
            board position. Each sub-list corresponds to a rank (row) on the board,
            ordered from 1 to 8, and has 8 elements (integers), each corresponding to
            a square on that rank, ordered from a to h. For example, `board[0][1]`
            corresponds to the square on first row (rank) and second column, i.e. b1.
            The data for each square is an integer from -6 to +6, where
            0: empty, 1: pawn, 2: knight, 3: bishop, 4: rook, 5: queen, 6: king
            White pieces are denoted with positive integers, while black pieces have the
            same magnitude but with a negative sign.
        castling_stats : list[list[int]]
            Castling availability for white and black, as a 2-dimensional list of integers.
            Each sub-list contains two integers, representing availability of the kingside
            and queenside castling, respectively. The first sub-list corresponds to white,
            and the second sub-list ro black. Integers are either 1 (castling allowed) or 0.
        turn : int
            The player who should play first, where +1 is white and -1 is black.
        fifty_move_count : int
            Number of plies (half-moves) since the last capture or pawn advance, used for
             the fifty-move-draw rule; if the number reaches 100, the game ends in a draw.
        enpassant_file : int
            The file (column) index (from 0 to 7), in which an en passant capture is allowed
            for the current player in the current move. Defaults to -1 if no en passant allowed.
        ply_count : int
            The number of plies (half-moves) from the beginning of the game. Starts at 0.
        """
        ...

    @classmethod
    def new(cls) -> Chessboard:
        """
        Instantiate a new Chessboard in the starting position of a standard game.

        Returns
        -------
        Chessboard
        """
        board = [
            [4, 2, 3, 5, 6, 3, 2, 4],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-4, -2, -3, -5, -6, -3, -2, -4]
        ]
        castling_stats = [[1, 1], [1, 1]]
        return cls(board=board, castling_stats=castling_stats)

    @classmethod
    def from_fen_record(cls, record: str) -> Chessboard:
        """
        Instantiate a new Chessboard from Forsyth–Edwards Notation (FEN) record.

        Parameters
        ----------
        record : str
            A FEN record as a string.

        Returns
        -------
        Chessboard
            A new Chessboard object with the given board position.
        """
        return cls(*algebraic_notation.parse_fen(record=record))

    @property
    @abstractmethod
    def current_player(self) -> Player:
        """
        First player to move.

        Returns
        -------
        Player
        """
        ...

    @property
    @abstractmethod
    def next_player(self) -> Player:
        """
        Second player to move.

        Returns
        -------
        Player
        """
        ...

    def
