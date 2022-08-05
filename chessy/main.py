import numpy as np
import pygame as pg
import sys

from time import time

from chessy.gui.main import *

from chessy.board_representation import BoardState, Move
from chessy.judges.abc import IllegalMoveError, GameOverError
from chessy.judges.square_list import ArrayJudge


def main():
    new_game_state = BoardState.create_new_game()
    judge = ArrayJudge.load_state(new_game_state)
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
                                judge.submit_move(
                                    Move(
                                        s0=np.array(move[0]),
                                        s1=np.array(move[1]),
                                    )
                                )
                            except IllegalMoveError as e:
                                print(e)
                            except GameOverError as e:
                                print(e)
                            move = []

        for square in surf_squares:
            screen.blit(source=square[1], dest=square[0])
        for y, row in enumerate(judge.board):
            for x, piece in enumerate(row):
                if piece != 0:
                    surf = surf_pieces[piece]
                    rect = surf.get_rect(
                        center=(
                            MARGIN_BOARD + (2 * x + 1) * DIM_SQUARE[0] / 2,
                            DIM_BOARD[1] - MARGIN_BOARD - (2 * y + 1) * DIM_SQUARE[1] / 2,
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


main()
