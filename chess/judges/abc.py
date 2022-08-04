"""
This module contains the abstract base class (abc) for all Chessboard classes.
"""

# Standard library
from __future__ import annotations
from typing import NamedTuple, Any, Tuple, Sequence, NoReturn, Optional
from abc import ABC, abstractmethod
# 3rd party
import numpy as np
# Self
from ..board_representation import BoardState, Move


class Judge(ABC):
    """

    """

    @abstractmethod
    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def load_state(cls, state: BoardState) -> Judge:
        """
        Instantiate a new Judge for a given board state.
        """
        ...

    @abstractmethod
    def reveal_current_state(self) -> BoardState:
        """
        Generate a BoardState representation for current internal state.
        """
        ...

    @property
    @abstractmethod
    def valid_moves(self) -> list[Move]:
        """
        Generate all valid moves for the current player.
        A move is represented as a new BoardState object.
        """

    @abstractmethod
    def submit_move(self, move: Move) -> NoReturn:
        """
        Apply a given move to the current state.

        Parameters
        ----------


        Raises
        ------
        IllegalMoveError
            When the move is illegal.
        """

    @property
    @abstractmethod
    def move_is_promotion(self) -> bool:
        ...

    @property
    @abstractmethod
    def board_is_checkmate(self) -> bool:
        """
        Whether the current player is checkmated.

        Returns
        -------
        bool
        """
        ...

    @property
    @abstractmethod
    def board_is_draw(self) -> bool:
        """
        Whether the game is a draw, due to the board configuration (not fifty move draw rule).

        Returns
        -------
        bool
        """
        ...


class IllegalMoveError(Exception):
    pass


class GameOverError(Exception):
    pass