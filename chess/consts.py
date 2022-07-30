"""
This module contains all the conventions used for colors, pieces and squares in the program.
"""

from typing import NamedTuple


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


COLORS = {-1: Color(name="black", letter="b"), 1: Color(name="white", letter="w")}

PIECES = {
    -6: Piece(color=COLORS[-1], name="king",   letter="K", symbol="♚"),
    -5: Piece(color=COLORS[-1], name="queen",  letter="Q", symbol="♛"),
    -4: Piece(color=COLORS[-1], name="rook",   letter="R", symbol="♜"),
    -3: Piece(color=COLORS[-1], name="bishop", letter="B", symbol="♝"),
    -2: Piece(color=COLORS[-1], name="knight", letter="N", symbol="♞"),
    -1: Piece(color=COLORS[-1], name="pawn",   letter="P", symbol="♟"),
    +1: Piece(color=COLORS[+1], name="pawn",   letter="P", symbol="♙"),
    +2: Piece(color=COLORS[+1], name="knight", letter="N", symbol="♘"),
    +3: Piece(color=COLORS[+1], name="bishop", letter="B", symbol="♗"),
    +4: Piece(color=COLORS[+1], name="rook",   letter="R", symbol="♖"),
    +5: Piece(color=COLORS[+1], name="queen",  letter="Q", symbol="♕"),
    +6: Piece(color=COLORS[+1], name="king",   letter="K", symbol="♔"),
}

FILES = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}

RANKS = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8"}

SQUARES = {
    (rank_idx, file_idx): f"{file}{rank}"
    for rank_idx, rank in RANKS.items() for file_idx, file in FILES.items()
}
