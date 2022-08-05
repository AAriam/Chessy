
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
            piece = turn * PIECE_LETTERS[move[0]]
        else:
            column = FILES[move[0]]
        print(f"{idx}: ({turn}) ")



