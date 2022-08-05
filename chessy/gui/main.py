from typing import NamedTuple, Union
import sys
import pygame as pg
import numpy as np

from pathlib import Path


class Gui:

    CAPTION_WIN = "CHESSY"
    POS_WHITE_SQUARES = [(i, j) for i in range(8) for j in range(1 if i % 2 == 0 else 0, 8, 2)]
    POS_BLACK_SQUARES = [(i, j) for i in range(8) for j in range(0 if i % 2 == 0 else 1, 8, 2)]

    def __init__(
        self,
        width_win: int = 1200,
        width_menu_percent: int = 30,
        width_board_percent: int = 85,
        color_background: tuple[int, int, int] = (70, 78, 84),
        color_board: tuple[int, int, int] = (54, 26, 3),
        color_white_square: tuple[int, int, int] = (255, 255, 204),
        color_black_square: tuple[int, int, int] = (76, 153, 0),
        color_font: tuple[int, int, int] = (255, 255, 255),
        piece_set: str = "standard",
        fps: int = 60,
    ):

        self._width_win = width_win
        self._width_menu_percent = width_menu_percent
        self._width_board_percent = width_board_percent
        self._color_background = color_background
        self._color_board = color_board
        self._color_white_square = color_white_square
        self._color_black_square = color_black_square
        self._color_font = color_font
        self._piece_set = piece_set
        self._fps = fps

        # Following attributes set in `self.calibrate`
        self._height: float = None
        self._dim_win: np.ndarray = None
        self._dim_board: np.ndarray = None
        self._dim_chessboard: np.ndarray = None
        self._margin_board: float = None
        self._dim_square: np.ndarray = None
        self._fontsize: int = None
        self.calibrate()

        self._surf_board = None
        self._screen = None
        self._font = None
        self.pieces = None

        pg.init()
        pg.display.set_caption(self.CAPTION_WIN)
        self._clock = pg.time.Clock()
        return

    @property
    def width_win(self):
        return self._width_win

    @width_win.setter
    def width_win(self, value):
        self._width_win = value

    @property
    def width_menu_percent(self):
        return self._width_menu_percent

    @width_menu_percent.setter
    def width_menu_percent(self, value):
        self._width_menu_percent = value

    @property
    def width_board_percent(self):
        return self._width_board_percent

    @width_board_percent.setter
    def width_board_percent(self, value):
        self._width_board_percent = value

    @property
    def color_background(self):
        return self._color_background

    @color_background.setter
    def color_background(self, value):
        self._color_background = value

    @property
    def color_board(self):
        return self._color_board

    @color_board.setter
    def color_board(self, value):
        self._color_board = value

    @property
    def color_white_square(self):
        return self._color_white_square

    @color_white_square.setter
    def color_white_square(self, value):
        self._color_white_square = value

    @property
    def color_black_square(self):
        return self._color_black_square

    @color_black_square.setter
    def color_black_square(self, value):
        self._color_black_square = value

    @property
    def color_font(self):
        return self._color_font

    @color_font.setter
    def color_font(self, value):
        self._color_font = value

    @property
    def piece_set(self):
        return self._piece_set

    @piece_set.setter
    def piece_set(self, value: int):
        self._piece_set = value

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value: int):
        self._fps = value

    def calibrate(self):
        self._height = (1 - self.width_menu_percent/100) * self.width_win
        self._dim_win = np.array([self.width_win, self._height])
        self._dim_board = np.array([self._height, self._height])
        self._dim_chessboard = self.width_board_percent * self._dim_board / 100
        self._margin_board = (self._dim_board[0] - self._dim_chessboard[0]) / 2
        self._dim_square = self._dim_chessboard // 8
        self._fontsize = self.width_win // 20  # 60
        return

    def create_ranks_files_labels(self):
        obj_rank_labels = []
        obj_file_labels = []
        for idx, label in enumerate("abcdefgh"):
            surf_file_label = self._font.render(label, True, self.color_font)
            rect_file_label = surf_file_label.get_rect(
                center=(
                    self._margin_board + (2 * idx + 1) * self._dim_square[0] / 2,
                    self._dim_board[1] - self._margin_board / 2,
                )
            )
            surf_rank_label = self._font.render(str(8 - idx), True, self.color_font)
            rect_rank_label = surf_rank_label.get_rect(
                center=(
                    self._margin_board / 2,
                    self._margin_board + (2 * idx + 1) * self._dim_square[1] / 2,
                )
            )
            obj_file_labels.append({"surf": surf_file_label, "rect": rect_file_label})
            obj_rank_labels.append({"surf": surf_rank_label, "rect": rect_rank_label})
        return obj_file_labels, obj_rank_labels

    def create_squares(self) -> list[dict[str, Union[pg.Surface, list[tuple[int, int]]]]]:
        positions = self.POS_WHITE_SQUARES + self.POS_BLACK_SQUARES  # Concatenate
        colors = [self.color_white_square] * 32 + [self.color_black_square] * 32  # Concatenate
        obj_squares = []
        for position, color in zip(positions, colors):
            surf_square = pg.Surface(size=self._dim_square)
            surf_square.fill(color=color)
            rect_square = surf_square.get_rect(
                topleft=(
                    position[1] * self._dim_square[0] + self._margin_board,
                    self._dim_board[0]
                    - self._margin_board
                    - (position[0] + 1) * self._dim_square[1],
                )
            )
            obj_squares.append(
                {"rect": rect_square, "surf": surf_square, "pos": position}
            )
        return obj_squares

    def create_pieces(self):
        surf_pieces = dict()
        for i in range(1, 7):
            for j in [i, -i]:
                surf = pg.transform.smoothscale(
                    pg.image.load(
                        Path.absolute(Path(f"gui/graphics/piece_sets/{self.piece_set}/{j}.png"))
                    ),
                    0.9 * self._dim_square,
                )
                surf_pieces[j] = surf
        return surf_pieces

    def draw_statics(self):
        self.calibrate()
        self._font = pg.font.Font(None, self._fontsize)
        self._screen = pg.display.set_mode(size=self._dim_win)  # Create a display surface
        self._screen.fill(color=self.color_background)
        self._surf_board = pg.Surface(size=self._dim_board)
        self._surf_board.fill(color=self.color_board)
        self.pieces = self.create_pieces()
        self.squares = self.create_squares()

        obj_file_labels, obj_rank_labels = self.create_ranks_files_labels()
        self._screen.blit(source=self._surf_board, dest=(0, 0))
        for file_label, rank_label in zip(obj_file_labels, obj_rank_labels):
            self._screen.blit(source=file_label["surf"], dest=file_label["rect"])
            self._screen.blit(source=rank_label["surf"], dest=rank_label["rect"])
        return
