from typing import NamedTuple, Union, Optional
import sys
import pygame as pg
import numpy as np

from pathlib import Path

from chessy.user_interface import GameInterface, Move
from chessy.consts import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK
from chessy.judges.abc import IllegalMoveError, GameOverError

class GraphicalInterface(GameInterface):

    CAPTION_WIN = "CHESSY"
    POS_WHITE_SQUARES = [(i, j) for i in range(8) for j in range(1 if i % 2 == 0 else 0, 8, 2)]
    POS_BLACK_SQUARES = [(i, j) for i in range(8) for j in range(0 if i % 2 == 0 else 1, 8, 2)]

    def __init__(
        self,
        num_players: int = 2,
        color_player: str = "white",
        initial_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        time_left_white: float = 0,
        time_left_black: float = 0,
        undo_allowed: bool = False,
    ):
        super().__init__(
            num_players=num_players,
            color_player=color_player,
            initial_state=initial_state,
            time_left_white=time_left_white,
            time_left_black=time_left_black,
            undo_allowed=undo_allowed
        )


        self.width_menu_percent: int = 30
        self.width_board_percent: int = 85
        self.color_background: tuple[int, int, int] = (70, 78, 84)
        self.color_board: tuple[int, int, int] = (54, 26, 3)
        self.color_white_square: tuple[int, int, int] = (255, 255, 204)
        self.color_black_square: tuple[int, int, int] = (76, 153, 0)
        self.color_highlighted_square: tuple[int, int, int] = (70, 70, 200)
        self.color_font: tuple[int, int, int] = (255, 255, 255)
        self.piece_set: str = "standard"
        self.fps: int = 60

        pg.init()
        info_display = pg.display.Info()
        width_display = info_display.current_w
        height_display = info_display.current_h
        self.width_win: int = width_display * 0.5

        # Following attributes set in `self.calibrate`
        self.height: float = None
        self.dim_win: np.ndarray = None
        self.dim_board: np.ndarray = None
        self.dim_chessboard: np.ndarray = None
        self.range_chessboard: np.ndarray = None
        self.margin_board: float = None
        self.dim_square: np.ndarray = None
        self.squares = None
        self.coord_squares_center: dict[tuple[int, int], tuple[float, float]] = None
        self.fontsize: int = None
        self.calibrate()

        self.surf_board = None
        self.screen = None
        self.font = None
        self.pieces = None
        self.s_selected: Optional[tuple] = None

        pg.display.set_caption(self.CAPTION_WIN)
        self.clock = pg.time.Clock()
        self.EVENT_ACTION_MAPPING = {
            pg.QUIT: self.quit_game,
            pg.MOUSEBUTTONUP: self.click

        }
        self.draw_statics()
        self.run()

        return

    def calibrate(self):
        self.height = (1 - self.width_menu_percent / 100) * self.width_win
        self.dim_win = np.array([self.width_win, self.height])
        self.dim_board = np.array([self.height, self.height])
        self.dim_chessboard = self.width_board_percent / 100 * self.dim_board
        self.margin_board = (self.dim_board[0] - self.dim_chessboard[0]) / 2
        self.dim_square = self.dim_chessboard // 8
        self.fontsize = int(self.width_win // 25)
        self.range_chessboard = np.array(
            [[self.margin_board, self.margin_board], self.dim_chessboard + self.margin_board]
        )
        self.coord_squares_center = self.coord_square_centers()
        return

    def draw_statics(self):
        self.calibrate()
        self.font = pg.font.Font(None, self.fontsize)
        self.screen = pg.display.set_mode(size=self.dim_win)  # Create a display surface
        self.screen.fill(color=self.color_background)
        self.surf_board = pg.Surface(size=self.dim_board)
        self.surf_board.fill(color=self.color_board)
        self.pieces = self.create_pieces()
        self.squares = self.create_squares()

        obj_file_labels, obj_rank_labels = self.create_ranks_files_labels()
        self.screen.blit(source=self.surf_board, dest=(0, 0))
        for file_label, rank_label in zip(obj_file_labels, obj_rank_labels):
            self.screen.blit(source=file_label["surf"], dest=file_label["rect"])
            self.screen.blit(source=rank_label["surf"], dest=rank_label["rect"])
        return

    def run(self):
        self.ss_selected = []
        while True:
            events = pg.event.get()
            for event in events:
                # print(event.type, event)
                action = self.EVENT_ACTION_MAPPING.get(event.type)
                if action:
                    action(event)
            self.redraw_board()
        return

    def redraw_board(self):
        for square in self.squares.values():
            self.screen.blit(source=square["surf"], dest=square["rect"])
        occupied_squares = self.game.judge.occupied_squares
        corresp_pieces = self.game.judge.pieces_in_squares(ss=occupied_squares)

        for y, row in enumerate(self.game.current_state.board):
            for x, piece in enumerate(row):
                if piece != 0:
                    surf = self.pieces[piece]
                    rect = surf.get_rect(
                        center=(
                            self.margin_board + (2 * x + 1) *
                            self.dim_square[0] / 2,
                            self.dim_board[1]
                            - self.margin_board
                            - (2 * y + 1) * self.dim_square[1] / 2,
                        )
                    )
                    self.screen.blit(source=surf, dest=rect)
        pg.display.update()
        self.clock.tick(self.fps)
        return

    def click(self, event):
        pos = np.array(event.pos)
        if self.position_is_inside_chessboard(pos):
            self.click_on_board(pos)

        file = np.int8((pos[0] - self.margin_board) // self.dim_square[0])
        rank = np.int8(
            7 - (pos[1] - self.margin_board) // self.dim_square[1])

        # print(rank, file)

    def click_on_board(self, pos):
        s = np.flip((pos - self.margin_board) // self.dim_square).astype(
            np.int8)
        s[0] = 7 - s[0]

        if self.s_selected is None:
            if self.game.judge.square_belongs_to(s=s, p=self.game.current_player):
                self.squares[tuple(s)]["surf"].fill(color=self.color_highlighted_square)
                self.s_selected = s
        else:
            square = self.squares[tuple(self.s_selected)]
            square["surf"].fill(color=square["color"])
            # square = self.squares[tuple(self.ss_selected[0])]
            # square["surf"].fill(color=square["color"])
            # for s in self.ss_selected:
            #     square = self.squares[tuple(s)]
            #     square["surf"].fill(color=square["color"])
            s0 = self.s_selected
            self.s_selected = None
            self.submit_move(s0, s)
        return

    def submit_move(self, s0, s1, pp=None):

        # for event in events:
        #     if event.type == pg.MOUSEBUTTONDOWN:
        #         for square in gui.squares:
        #             if square["rect"].collidepoint(event.pos):
        #                 selected_squares.append(square)
        #                 square["surf"].fill(color=(50, 50, 162, 50))
        #                 move.append(square["pos"])
        #                 for valid_move in judge.valid_moves:
        #                     if np.all(valid_move.s0 == square["pos"]):
        #                         pass
        #                 if len(move) == 2:
        #                     for selected_square in selected_squares:
        #
        #                         selected_square["surf"].fill(color=selected_square["color"])
        #                     try:
        #                         judge.submit_move(
        #                             Move(
        #                                 s0=np.array(move[0]),
        #                                 s1=np.array(move[1]),
        #                                 p=judge.pieces_in_squares(np.array(move[0])),
        #                             )
        #                         )
        #                     except IllegalMoveError as e:
        #                         print(e)
        #                     except GameOverError as e:
        #                         print(e)
        #                     except Exception as e:
        #                         print(e)
        #                     move = []
        try:
            self.game.submit_move(
                Move(s0=s0, s1=s1)
            )
        except IllegalMoveError as e:
            print(e)
        except GameOverError as e:
            print(e)
        else:
            pass
        finally:
            self.ss_selected = []
        return

    def position_is_inside_chessboard(self, pos):
        return np.all(pos > self.range_chessboard[0]) and np.all(
            pos < self.range_chessboard[1]
        )

    def create_ranks_files_labels(self):
        obj_rank_labels = []
        obj_file_labels = []
        for idx, label in enumerate("abcdefgh"):
            surf_file_label = self.font.render(label, True, self.color_font)
            rect_file_label = surf_file_label.get_rect(
                center=(
                    self.margin_board + (2 * idx + 1) * self.dim_square[0] / 2,
                    self.dim_board[1] - self.margin_board / 2,
                )
            )
            surf_rank_label = self.font.render(str(8 - idx), True, self.color_font)
            rect_rank_label = surf_rank_label.get_rect(
                center=(
                    self.margin_board / 2,
                    self.margin_board + (2 * idx + 1) * self.dim_square[1] / 2,
                )
            )
            obj_file_labels.append({"surf": surf_file_label, "rect": rect_file_label})
            obj_rank_labels.append({"surf": surf_rank_label, "rect": rect_rank_label})
        return obj_file_labels, obj_rank_labels

    def create_squares(
            self
    ) -> dict[tuple[int, int], dict[str, Union[pg.Rect, pg.Surface, tuple]]]:
        """
        Create `Surface` objects for squares.

        Returns
        -------
        dict[tuple[int, int], dict[str, Union[pg.Rect, pg.Surface, tuple]]]
            A dictionary where keys are internal coordinates of squares (e.g. (1, 0)), and values
            are dictionaries with keys `rect`, `surf` and `color` corresponding to the `Surface`
            object of that square.
        """
        positions = self.POS_WHITE_SQUARES + self.POS_BLACK_SQUARES  # Concatenate
        colors = [self.color_white_square] * 32 + [self.color_black_square] * 32  # Concatenate
        squares = dict()
        for position, color in zip(positions, colors):
            surf_square = pg.Surface(size=self.dim_square)
            surf_square.fill(color=color)
            rect_square = surf_square.get_rect(
                center=self.coord_squares_center[position]
            )
            squares[position] = {
                "rect": rect_square,
                "surf": surf_square,
                "color": color
            }
        return squares

    def create_pieces(self) -> dict[int, pg.Surface]:
        """
        Load piece-set images from file and create Surface objects.

        Returns
        -------
        dict[int, pg.Surface]
            A dictionary where keys are piece-IDs, and values are corresponding `Surface` objects.
        """
        surf_pieces = dict()
        for p in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
            for c in [WHITE, BLACK]:
                surf = pg.transform.smoothscale(
                    pg.image.load(
                        Path.absolute(Path(f"gui/graphics/piece_sets/{self.piece_set}/{c*p}.png"))
                    ),
                    0.9 * self.dim_square,
                )
                surf_pieces[c*p] = surf
        return surf_pieces

    def coord_square_centers(self) -> dict[tuple[int, int], tuple[float, float]]:
        """
        Calculate the coordinates of the center of each square, according to current's window size.

        Returns
        -------
        dict[tuple[int, int], tuple[float, float]]
            A dictionary where keys are internal coordinates of squares (e.g. (1, 0)), and values
            are corresponding coordinates on the window.
        """
        return {
            (i, j): (
                self.margin_board + (j + 0.5) * self.dim_square[0],
                self.dim_board[1] - self.margin_board - (i + 0.5) * self.dim_square[1]
            ) for i in range(8) for j in range(8)
        }


    @staticmethod
    def quit_game(event):
        pg.quit()
        sys.exit()
