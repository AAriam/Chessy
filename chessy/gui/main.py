import sys
import pygame as pg
import numpy as np

from pathlib import Path


FRAMERATE = 60
WIDTH_MAX = 1200  # Max. width of main window
WIDTH_PERCENTAGE_MENU = 0.3
WIDTH_PERCENTAGE_CHESSBOARD = 0.85

COLOR_BLACK_SQUARE = (76, 153, 0)
COLOR_WHITE_SQUARE = (255, 255, 204)
COLOR_BACKGROUND = (70, 78, 84)
COLOR_BOARD = (54, 26, 3)

FONTSIZE = 60

CAPTION_WIN = "CHESSY"


HEIGHT = (1 - WIDTH_PERCENTAGE_MENU) * WIDTH_MAX
DIM_WIN = np.array([WIDTH_MAX, HEIGHT])
DIM_BOARD = np.array([HEIGHT, HEIGHT])
DIM_CHESSBOAR = WIDTH_PERCENTAGE_CHESSBOARD * DIM_BOARD
MARGIN_BOARD = (DIM_BOARD[0] - DIM_CHESSBOAR[0]) / 2
DIM_SQUARE = DIM_CHESSBOAR // 8

WHITE_SQUARE_POS = [(i, j) for i in range(8) for j in range(1 if i % 2 == 0 else 0, 8, 2)]
BLACK_SQUARE_POS = [(i, j) for i in range(8) for j in range(0 if i % 2 == 0 else 1, 8, 2)]


pg.init()
pg.display.set_caption(CAPTION_WIN)
clock = pg.time.Clock()
font = pg.font.Font(None, FONTSIZE)


screen = pg.display.set_mode(size=DIM_WIN)  # Create a display surface
screen.fill(color=COLOR_BACKGROUND)

surf_board = pg.Surface(size=DIM_BOARD)
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
                MARGIN_BOARD + (2 * idx + 1) * DIM_SQUARE[0] / 2,
                DIM_BOARD[1] - MARGIN_BOARD / 2,
            )
        )
    )
    surf = font.render(str(8 - idx), True, "White")
    surf_row_labels.append(surf)
    rect_row_labels.append(
        surf.get_rect(center=(MARGIN_BOARD / 2, MARGIN_BOARD + (2 * idx + 1) * DIM_SQUARE[1] / 2))
    )


surf_squares = []
for pos in BLACK_SQUARE_POS:
    surf = pg.Surface(size=DIM_SQUARE)
    surf.fill(color=COLOR_BLACK_SQUARE)
    rect = surf.get_rect(
        topleft=(
            pos[1] * DIM_SQUARE[0] + MARGIN_BOARD,
            DIM_BOARD[0] - MARGIN_BOARD - (pos[0] + 1) * DIM_SQUARE[1],
        )
    )
    surf_squares.append([rect, surf, pos])
for pos in WHITE_SQUARE_POS:
    surf = pg.Surface(size=DIM_SQUARE)
    surf.fill(color=COLOR_WHITE_SQUARE)
    rect = surf.get_rect(
        topleft=(
            pos[1] * DIM_SQUARE[0] + MARGIN_BOARD,
            DIM_BOARD[0] - MARGIN_BOARD - (pos[0] + 1) * DIM_SQUARE[1],
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
            pg.image.load(Path.absolute(Path(f"gui/graphics/pieces/{j}.png"))), 0.9 * DIM_SQUARE
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


# from game import ChessGame, IllegalMoveError

# game = ChessGame()
