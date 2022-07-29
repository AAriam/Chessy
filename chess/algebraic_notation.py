
example = "1.e4 c5 2.Nf3 a6 3.c3 d5 4.exd5 Qxd5 5.d4 Nf6 6.Be2 e6 7.O-O Be7 8.c4 Qd8 9.Nc3 O-O 10.dxc5 Bxc5 11.a3 Be7 12.Be3 Nbd7 13.Nd4 Qc7 14.h3 b6 15.b4 Bb7 16.Qb3 Rac8 17.Rac1 Bd6 18.Rfd1 Bh2+ 19.Kh1 Bf4"


FILES = {file: idx for idx, file in enumerate("abcdefgh")}  # Columns
RANKS = {idx: idx - 1 for idx in range(1, 9)}  # Rows

PIECES = {letter: idx + 1 for idx, letter in enumerate("pNBRQK")}
PIECES_SYMBOLS_BLACK = {symbol: idx - 6 for idx, symbol in enumerate("♚♛♜♝♞♟")}
PIECES_SYMBOLS_WHITE = {symbol: idx + 1 for idx, symbol in enumerate("♙♘♗♖♕♔")}
PIECES_SYMBOLS = PIECES_SYMBOLS_BLACK | PIECES_SYMBOLS_WHITE


def read(notation: str):
    game = [entry.split(".")[-1] for entry in notation.split()]
    for idx, move in enumerate(game):
        piece=None
        column=None
        turn = (-1) ** idx
        if move == "0-0":
            pass
        elif move == "0-0-0":
            pass
        elif move[0] == move[0].upper():
            piece = turn * PIECES[move[0]]
        else:
            column = FILES[move[0]]
        print(f"{idx}: ({turn}) ")
read(example)