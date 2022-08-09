from abc import ABC, abstractmethod

import numpy as np
import pygame as pg
import sys


from time import time

from chessy.board_representation import BoardState, Move
from chessy.judges.abc import IllegalMoveError, GameOverError
from chessy.judges.square_list import ArrayJudge
from chessy.notations import fen
from chessy.game import Game


class GameInterface(ABC):
    def __init__(
        self,
        num_players: int = 2,
        color_player: str = "white",
        initial_state: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        time_left_white: float = 0,
        time_left_black: float = 0,
        undo_allowed: bool = False,
    ):
        self.game = Game(
            initial_state=fen.to_boardstate(initial_state),
            time_left_white=time_left_white,
            time_left_black=time_left_black
        )
        self.undo_allowed = undo_allowed
