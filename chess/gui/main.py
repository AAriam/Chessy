import sys
import pygame as pg
import numpy as np


FRAMERATE = 60
WIDTH_MAX = 1500  # Max. width of main window
WIDTH_PERCENTAGE_MENU = 0.3
WIDTH_PERCENTAGE_CHESSBOARD = 0.85

COLOR_BLACK_SQUARE = (76, 153, 0)
COLOR_WHITE_SQUARE = (255, 255, 204)
COLOR_BACKGROUND = (70, 78, 84)
COLOR_BOARD = (54, 26, 3)


height = (1 - WIDTH_PERCENTAGE_MENU) * WIDTH_MAX
dim_win = np.array([WIDTH_MAX, height])
dim_board = np.array([height, height])
dim_chessboard = WIDTH_PERCENTAGE_CHESSBOARD * dim_board
margin_board = (dim_board[0] - dim_chessboard[0]) / 2
dim_square = dim_chessboard // 8

WHITE_SQUARE_POS = [(i, j) for i in range(8) for j in range(1 if i % 2 == 0 else 0, 8, 2)]
BLACK_SQUARE_POS = [(i, j) for i in range(8) for j in range(0 if i % 2 == 0 else 1, 8, 2)]


pg.init()
pg.display.set_caption("Chess")
clock = pg.time.Clock()
font = pg.font.Font(None, 60)


screen = pg.display.set_mode(size=dim_win)  # Create a display surface
screen.fill(color=COLOR_BACKGROUND)

surf_board = pg.Surface(size=dim_board)
surf_board.fill(color=COLOR_BOARD)


# Draw row and column labels
surf_column_labels = []
rect_column_labels = []
surf_row_labels = []
rect_row_labels = []
for idx, label in enumerate("abcdefgh"):
    surf = font.render(label, True, "White")
    surf_column_labels.append(surf)
    rect_column_labels.append(
        surf.get_rect(
            center=(
                margin_board + (2 * idx + 1) * dim_square[0] / 2,
                dim_board[1] - margin_board / 2,
            )
        )
    )
    surf = font.render(str(8 - idx), True, "White")
    surf_row_labels.append(surf)
    rect_row_labels.append(
        surf.get_rect(center=(margin_board / 2, margin_board + (2 * idx + 1) * dim_square[1] / 2))
    )


surf_squares = []
for pos in BLACK_SQUARE_POS:
    surf = pg.Surface(size=dim_square)
    surf.fill(color=COLOR_BLACK_SQUARE)
    rect = surf.get_rect(
        topleft=(
            pos[1] * dim_square[0] + margin_board,
            dim_board[0] - margin_board - (pos[0] + 1) * dim_square[1],
        )
    )
    surf_squares.append([rect, surf, pos])
for pos in WHITE_SQUARE_POS:
    surf = pg.Surface(size=dim_square)
    surf.fill(color=COLOR_WHITE_SQUARE)
    rect = surf.get_rect(
        topleft=(
            pos[1] * dim_square[0] + margin_board,
            dim_board[0] - margin_board - (pos[0] + 1) * dim_square[1],
        )
    )
    surf_squares.append([rect, surf, pos])

# surf_white_square = pg.Surface(size=dim_square)
# surf_white_square.fill(color=COLOR_WHITE_SQUARE)
#
# surf_black_square = pg.Surface(size=dim_square)
# surf_black_square.fill(color=COLOR_BLACK_SQUARE)


surf_pieces = dict()
for i in range(1, 7):
    for j in [i, -i]:
        surf = pg.transform.smoothscale(
            pg.image.load(f"graphics/pieces/{j}.png"), 0.9 * dim_square
        )
        surf_pieces[j] = surf


screen.blit(source=surf_board, dest=(0, 0))

# for pos in BLACK_SQUARE_POS:
#     screen.blit(source=surf_black_square, dest=(pos[1] * dim_square[0] + margin_board, dim_board[0] - margin_board - (pos[0]+1) * dim_square[1]))
# for pos in WHITE_SQUARE_POS:
#     screen.blit(source=surf_white_square, dest=(pos[1] * dim_square[0] + margin_board, dim_board[0] - margin_board - (pos[0]+1) * dim_square[1]))


for i in range(8):
    screen.blit(source=surf_column_labels[i], dest=rect_column_labels[i])
    screen.blit(source=surf_row_labels[i], dest=rect_row_labels[i])


from game import ChessGame, IllegalMoveError

game = ChessGame()

move = []

while True:
    counter = 0
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.MOUSEBUTTONDOWN:
            for rect, surf, pos in surf_squares:
                if rect.collidepoint(event.pos):
                    move.append(pos)
                    if len(move) == 2:
                        try:
                            game.move(np.array(move[0]), np.array(move[1]))
                        except IllegalMoveError as e:
                            print(e)
                        move = []

    for square in surf_squares:
        screen.blit(source=square[1], dest=square[0])
    for y, row in enumerate(game.board):
        for x, piece in enumerate(row):
            if piece != 0:
                surf = surf_pieces[piece]
                rect = surf.get_rect(
                    center=(
                        margin_board + (2 * x + 1) * dim_square[0] / 2,
                        dim_board[1] - margin_board - (2 * y + 1) * dim_square[1] / 2,
                    )
                )
                screen.blit(source=surf, dest=rect)
            # mouse_pos = pg.mouse.get_pos()
            # if rect.collidepoint(mouse_pos):
            #     print(y, x)
            #     if(pg.mouse.get_pressed()[0]):
            #         screen.blit(surf, mouse_pos)

    pg.display.update()
    clock.tick(FRAMERATE)
