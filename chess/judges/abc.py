"""
This module contains the abstract base class (abc) for all Chessboard classes.
"""

# Standard library
from __future__ import annotations
from typing import NamedTuple, Any, Tuple, Sequence
from abc import ABC, abstractmethod
# 3rd party
import numpy as np
# Self
from ..board_representation import BoardState


class Chessboard(ABC):
    """

    """

    @abstractmethod
    def __init__(self):
        ...

    @classmethod
    @abstractmethod
    def from_board_state(cls, board_state: BoardState) -> Chessboard:
        """
        Create a new board from internal board representation model.
        """
        ...

    @abstractmethod
    def to_board_state(self) -> BoardState:
        """
        Transform board into internal board representation model.
        """
        ...

    @abstractmethod
    def generate_all_valid_moves(self) -> list[BoardState]:
        """
        Generate all valid moves for the current player.
        A move is represented as a new BoardState object.
        """

    @abstractmethod
    def apply_move(
            self, s0: Sequence[int, int], s1: Sequence[int, int], promote_to: int
    ) -> BoardState:
        """
        Apply a given move.

        Parameters
        ----------
        s0 :
        s1 :
        promote_to : int

        Raises
        ------
        IllegalMoveError
            When the move is illegal.
        """

    @abstractmethod
    def move_is_promotion(self, s0: Sequence[int, int], s1: Sequence[int, int]) -> bool:
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

