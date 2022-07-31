"""
This module contains the data structures and conventions used in the whole program.
"""

# Standard library
from typing import NamedTuple, Optional
# 3rd party
import numpy as np


class BoardState(NamedTuple):
    """
    A data structure representing a full description of a chess position, i.e. the position state.

    board : numpy.ndarray(shape=(8, 8), dtype=numpy.int8)
            A 2-dimensional array representing a specific board position,
            i.e. the location of each piece on the board.
            Axis 0 (2nd. dimension) corresponds to files (columns) ordered from 'a' to 'h'.
            Axis 1 (1st. dimension) corresponds to ranks (rows) ordered from 1 to 8.
            Thus, indexing the board as `board[i, j]` gives the data for the square on row 'i'
            and column 'j'. For example, `board[0, 1]` is square 'b1'.
            The data for each square is an integer from –6 to +6, where:
            0 = empty, 1 = pawn, 2 = knight, 3 = bishop, 4 = rook, 5 = queen, 6 = king
            White pieces are denoted with positive integers, while black pieces have the
            same magnitude but with a negative sign (e.g. +6 = white king, –6 = black king).
    castling_rights : numpy.ndarray(shape=(2, 2), dtype=numpy.int8)
        A 2-dimensional array representing castling availability for white and black, i.e. whether
        either player is permanently disqualified to castle, both kingside and queenside.
        Axis 0 (2nd. dimension) corresponds to kingside and queenside availability.
        Axis 1 (1st. dimension) corresponds to white and black players.
        Data is a boolean integer: either 1 (castling allowed) or 0 (not allowed).
    player : numpy.int8
        Current player to move; +1 is white and -1 is black.
    enpassant_file : numpy.int8
        The file (column) index (from 0 to 7), in which an en passant capture is allowed
        for the current player in the current move. Defaults to -1 if no en passant allowed.
    fifty_move_count : numpy.int8
        Number of plies (half-moves) since the last capture or pawn advance, used for
        the fifty-move-draw rule; if the number reaches 100, the game ends in a draw.
    ply_count : numpy.int16
        The number of plies (half-moves) from the beginning of the game. Starts at 0.
    """
    board: np.ndarray
    castling_rights: np.ndarray
    player: np.int8
    enpassant_file: np.int8
    fifty_move_count: np.int8
    ply_count: np.int16


class Move(NamedTuple):
    """
    A data structure representing a move in the game.

    start_square : numpy.ndarray
        Row and column index of the start square (both from 0 to 7), respectively.
    end_square : Sequence[int, int]
        Row and column index of the end square (both from 0 to 7), respectively.
    promote_to : Optional[int]
        Piece number to promote a pawn into, when the move is a promotion.
    """
    start_square: np.ndarray
    end_square: np.ndarray
    promote_to: Optional[np.int8] = None


class Color(NamedTuple):
    name: str
    letter: str


class Piece(NamedTuple):
    color: Color
    name: str
    letter: str
    symbol: str


class Square(NamedTuple):
    file: str
    rank: int


COLOR = {-1: Color(name="black", letter="b"), 1: Color(name="white", letter="w")}
PIECE = {
    -6: Piece(color=COLOR[-1], name="king", letter="K", symbol="♚"),
    -5: Piece(color=COLOR[-1], name="queen", letter="Q", symbol="♛"),
    -4: Piece(color=COLOR[-1], name="rook", letter="R", symbol="♜"),
    -3: Piece(color=COLOR[-1], name="bishop", letter="B", symbol="♝"),
    -2: Piece(color=COLOR[-1], name="knight", letter="N", symbol="♞"),
    -1: Piece(color=COLOR[-1], name="pawn", letter="P", symbol="♟"),
    +1: Piece(color=COLOR[+1], name="pawn", letter="P", symbol="♙"),
    +2: Piece(color=COLOR[+1], name="knight", letter="N", symbol="♘"),
    +3: Piece(color=COLOR[+1], name="bishop", letter="B", symbol="♗"),
    +4: Piece(color=COLOR[+1], name="rook", letter="R", symbol="♖"),
    +5: Piece(color=COLOR[+1], name="queen", letter="Q", symbol="♕"),
    +6: Piece(color=COLOR[+1], name="king", letter="K", symbol="♔"),
}
FILE = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}
RANK = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8"}
SQUARE = {
    (rank_idx, file_idx): f"{file}{rank}"
    for rank_idx, rank in RANK.items() for file_idx, file in FILE.items()
}
CASTLE = {0: "kingside", 1: "queenside"}
