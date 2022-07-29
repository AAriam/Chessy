

FILES = {file: idx for idx, file in enumerate("abcdefgh")}  # Columns
RANKS = {idx: idx - 1 for idx in range(1, 9)}  # Rows

PIECES = {letter: idx + 1 for idx, letter in enumerate("pNBRQK")}
PIECES_SYMBOLS_BLACK = {symbol: idx - 6 for idx, symbol in enumerate("♚♛♜♝♞♟")}
PIECES_SYMBOLS_WHITE = {symbol: idx + 1 for idx, symbol in enumerate("♙♘♗♖♕♔")}
PIECES_SYMBOLS = PIECES_SYMBOLS_BLACK | PIECES_SYMBOLS_WHITE

