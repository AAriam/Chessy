"""
This module contains the mappings used in generating/parsing different notations.
"""

FILES = {file: idx for idx, file in enumerate("abcdefgh")}  # Columns
RANKS = {idx: idx - 1 for idx in range(1, 9)}  # Rows

PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
PIECE_LETTERS = {letter: idx + 1 for idx, letter in enumerate("PNBRQK")}

PIECES_SYMBOLS_BLACK = {symbol: idx - 6 for idx, symbol in enumerate("")}
PIECES_SYMBOLS_WHITE = {symbol: idx + 1 for idx, symbol in enumerate("")}
PIECES_SYMBOLS = PIECES_SYMBOLS_BLACK | PIECES_SYMBOLS_WHITE
