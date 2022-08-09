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
from ..board_representation import BoardState, Move, Moves, COLOR, PIECE


class Judge(ABC):
    """ """

    @abstractmethod
    def __init__(self, initial_state: BoardState):
        """
        Instantiate a new Judge for a given board state.
        """
        ...

    @property
    @abstractmethod
    def current_state(self) -> BoardState:
        """
        Generate a BoardState representation for current internal state.
        """
        ...

    @property
    @abstractmethod
    def valid_moves(self) -> Moves:
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
    CODES = {
        0: "Start-square is out of board.",
        1: "Start-square is empty.",
        2: "It is {player_name}'s turn.",
        3: "End-square is out of board.",
        4: "Start and end-square are the same.",
        5: "{piece_name}s cannot move in direction {move_vect}.",
        6: "Move does not resolve check.",
        7: "Submitted move is illegal.",
    }

    def __init__(
            self, code: int,
            player: Optional[int] = None,
            piece_type: Optional[int] = None,
            move_vect: Optional[np.ndarray] = None
    ):
        self.code = code
        kwargs = dict()
        if player is not None:
            kwargs["player_name"] = COLOR[player].name
        if piece_type is not None:
            kwargs["piece_name"] = PIECE[piece_type].name.capitalize()
        if move_vect is not None:
           kwargs["move_vect"] = move_vect
        self.message = self.CODES[code].format(**kwargs)
        super().__init__(self.message)
        return


class GameOverError(Exception):

    CODES = {
        1: "Game over. The initial board is faulty; The opponent is already checkmated.",
        -1: "Game over. Current player is checkmated.",
        0: "Game over. It is a draw."
    }

    def __init__(self, code: int):
        self.code = code
        self.message = self.CODES[code]
        super().__init__(self.message)
        return
