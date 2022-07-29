
example = "1.e4 c5 2.Nf3 a6 3.c3 d5 4.exd5 Qxd5 5.d4 Nf6 6.Be2 e6 7.O-O Be7 8.c4 Qd8 9.Nc3 O-O 10.dxc5 Bxc5 11.a3 Be7 12.Be3 Nbd7 13.Nd4 Qc7 14.h3 b6 15.b4 Bb7 16.Qb3 Rac8 17.Rac1 Bd6 18.Rfd1 Bh2+ 19.Kh1 Bf4"


FILES = {file: idx for idx, file in enumerate("abcdefgh")}  # Columns
RANKS = {idx: idx - 1 for idx in range(1, 9)}  # Rows

PIECES = {letter: idx + 1 for idx, letter in enumerate("PNBRQK")}
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


def parse_fen(record: str) -> tuple[list[list[int]], list[list[int]], int, int, int, int]:
    """
    Parse a Forsyth–Edwards Notation (FEN) record to transform the data into a format accepted
    by the class `boards.abc.Chessboard`.

    Parameters
    ----------
    record : str
        A FEN record as a string.

    Returns
    -------
    tuple[list[list[int]], list[list[int]], int, int, int, int]
        Board, castling status, current player (turn), fifty-move count, en passant file,
        and ply count, respectively. For more information, see `boards.abc.Chessboard`.
    """
    if not isinstance(record, str):
        raise ValueError("FEN record must be a string.")
    # A record contains six fields, each separated by a space:
    try:
        (
            piece_data,
            active_color,
            castling_avail,
            enpassant_square,
            halfmove_clock,
            fullmove_num
        ) = record.split()
    except ValueError:
        raise ValueError("FEN record must contain six fields, each separated by a space.")
    # 1. Piece Data
    # describes each rank (from 8 to 1), with a '/' separating ranks:
    ranks = (piece_data.split("/"))
    if len(ranks) != 8:
        raise ValueError("FEN piece data should contain eight ranks, each separated by a '/'.")
    ranks.reverse()
    # Within each rank, squares are described from file a to h. Each piece is identified
    # by its algebraic notation, while white pieces are designated using uppercase letters
    # and black pieces use lowercase letters. A set of one or more consecutive empty squares
    # within a rank is denoted by a digit from "1" to "8".
    board = []
    for rank in ranks:
        board.append([])
        for square in rank:
            if square.isnumeric():
                board[-1].append([0] * int(square))
            else:
                color = square.isupper()
                try:
                    board[-1].append(color * PIECES[square.upper()])
                except KeyError:
                    raise ValueError(f"Piece notation {square} is unknown.")
        if len(board[-1]) != 8:
            raise ValueError(f"Each rank should describe eight squares; got {len(board[-1])}")
    # 2. Active Color
    # "w" means that White is to move; "b" means that Black is to move.
    if active_color == "w":
        turn = 1
    elif active_color == "b":
        turn = -1
    else:
        raise ValueError(f"Active color must be either 'w' or 'b'; got {active_color}.")
    # 3. Castling Availability
    # If neither side has the ability to castle, this field uses the character "-".
    # Otherwise, it contains one or more letters:
    # Uppercase for white, lowercase for black; 'k' for kingside, 'q' for queenside.
    castling_stats = [[0, 0], [0, 0]]
    for avail in castling_avail:
        idx = 0 if avail.isupper() else 1
        if avail == "-":
            break
        elif avail.upper() == "K":
            castling_stats[idx][0] = 1
        elif avail.upper() == "Q":
            castling_stats[idx][1] = 1
        else:
            raise ValueError(f"Castling availability field unrecognized; got {avail}")
    # 4. En passant target square
    # over which a pawn has just passed while moving two squares; in algebraic notation.
    # If there is no en passant target square, this field uses the character "-".
    # This is recorded regardless of whether there is a pawn in position to capture en passant.
    if enpassant_square == "-":
        enpassant_file = -1
    else:
        try:
            enpassant_file = FILES[enpassant_square[0]]
        except KeyError:
            raise ValueError(
                f"En passant target square not recognized; got {enpassant_square}"
            )
    # 5. Halfmove clock
    # number of halfmoves since the last capture or pawn advance, used for the fifty-move rule.
    if isinstance(halfmove_clock, int) and (0 <= halfmove_clock <= 100):
        fifty_move_count = halfmove_clock
    else:
        raise ValueError(
            f"Halfmove clock must be an integer between 0 and 50; got {halfmove_clock}."
        )
    # 6. Fullmove number
    # The number of the full moves. It starts at 1 and is incremented after Black's move.
    # According to
    #   Bonsdorff et al., Schach und Zahl. Unterhaltsame Schachmathematik. pp. 11–13,
    # the longest-possible game lasts 5899 moves (i.e. 11798 plies).
    if isinstance(fullmove_num, int) and (1 <= halfmove_clock <= 5899):
        ply_count = (fullmove_num - 1) * 2 + (1 if turn == -1 else 0)
    else:
        raise ValueError(
            f"Fullmove number must be an integer between 1 and 5899; got {fullmove_num}."
        )
    return board, castling_stats, turn, fifty_move_count, enpassant_file, ply_count
