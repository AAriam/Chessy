from typing import Sequence, Tuple, Optional
import numpy as np


class ChessGame:

    _COLORS = {-1: "black", 1: "white"}
    _PIECES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}

    def __init__(self):
        # Set instance attributes describing the game state to their initial values
        self._board: np.ndarray = self.new_board()  # Chessboard in starting position
        self._turn: int = 1  # Whose turn it is; 1 for white, -1 for black
        self._can_castle: np.ndarray = np.array([[0, 0], [1, 1], [1, 1]], dtype=np.int8)
        self._enpassant: int = -1  # Column where en passant capture is allowed in next move
        self._fifty_move_draw_count: int = 0  # Count for the fifty move draw rule
        # Set other useful instance attributes
        self._game_over: bool = False  # Whether the game is over
        self._score: int = 0  # Score of white at the end of the game
        return

    @property
    def board(self) -> np.ndarray:
        """
        The chessboard, as a 2d-array of shape (8, 8), where axis 0 corresponds to ranks
        (i.e. rows) from 1 to 8, and axis 1 corresponds to files (i.e. columns) from 'a'
        to 'h'. Each element thus corresponds to a square, e.g. `board[0, 0]` corresponds
        to square 'a1', `board[0, 7]` to 'h1', and `board[7, 7]` to 'h8'.
        The elements are of type `numpy.byte`, and contain information about that square:
        0: empty, 1: pawn, 2: knight, 3: bishop, 4: rook, 5: queen, 6: king
        White pieces are denoted with positive integers, while black pieces have the
        same magnitude but with a negative sign.
        """
        return self._board

    @property
    def turn(self) -> int:
        """
        Whose turn it is to move, described as an integer:
        +1 for white, and -1 for black
        """
        return self._turn

    @property
    def can_castle(self) -> np.ndarray:
        """
        Castle allowance for white and black, as a 2d-array of shape (2, 2).
        Axis 0 corresponds to the white and black players, respectively, and
        axis 1 corresponds to kingside and queenside castles, respectively.
        Thus, e.g. `can_castle[0, 0]` corresponds to white's kingside castle.
        Each element is either 1 or 0, where 1 means allowed and 0 means not allowed.
        """
        return self._can_castle[1:]

    @property
    def enpassant(self) -> int:
        """
        Column index of one of opponent's pawns that can be captured en passant in the next move.
        If no en passant capture is possible, it defaults to -1.
        """
        return self._enpassant

    @property
    def fifty_move_draw_count(self) -> int:
        """
        Number of current non-interrupted plies (half-moves), in which no capture has been made
        and no pawn has been moved. If it reaches 100, the game ends in a draw.
        """
        return self._fifty_move_draw_count

    @property
    def game_over(self) -> bool:
        """
        Whether the game is over.
        """
        return self._game_over

    @property
    def score(self) -> int:
        """
        Score of white at the end of the game.
        0 for draw, 1 for win, and -1 for loss.
        Before the game is over, the value defaults to 0.
        """
        return self._score

    def move(self, s0: Tuple[int, int], s1: Tuple[int, int]) -> Optional[int]:
        """
        Make a move for the current player.

        Parameters
        ----------
        s0 : Sequence[int, int]
            Row and column index of the start square (both from 0 to 7), respectively.
        s1 : Sequence[int, int]
            Row and column index of the end square (both from 0 to 7), respectively.

        Returns
        -------
        Optional[int]
            The score of white player if the game is over, `None` otherwise.
        """
        if self._game_over:  # Return white's score if the game is over
            return self._score
        self.raise_for_illegal_move(s0=s0, s1=s1)  # Otherwise, raise an error if move is illegal
        self._update_game_state(s0=s0, s1=s1)  # Otherwise, apply the move and update state
        if self._game_over:  # Return white's score if the game is over after the move
            return self._score
        return

    def raise_for_illegal_move(self, s0: Tuple[int, int], s1: Tuple[int, int]) -> None:
        """
        Raise an IllegalMoveError if the given move is illegal.

        Parameters
        ----------
        s0 : Sequence[int, int]
            Row and column index of the start square (both from 0 to 7), respectively.
        s1 : Sequence[int, int]
            Row and column index of the end square (both from 0 to 7), respectively.

        Raises
        ------
        IllegalMoveError
            When the move is not legal.
        """
        # For moves starting from an empty square
        if self.square_is_empty(s=s0):
            raise IllegalMoveError("Start-square is empty.")
        # For wrong turn (i.e. it is one player's turn, but other player's piece is being moved)
        if not self.square_belongs_to_current_player(s=s0):
            raise IllegalMoveError(f"It is {self._COLORS[self._turn]}'s turn.")
        # For move ending in a square occupied by current player's own pieces
        if self.square_belongs_to_current_player(s=s1):
            raise IllegalMoveError("End-square occupied by current player's piece.")
        # For move beginning or leading outside the board
        if self.square_is_outside_board(s=s0):
            raise IllegalMoveError("Start-square is out of board.")
        if self.square_is_outside_board(s=s1):
            raise IllegalMoveError("End-square is out of board.")
        return

    def square_is_empty(self, s: Tuple[int, int]) -> bool:
        """
        Whether a given square is empty.
        """
        return self._board[s] == 0

    def square_belongs_to_current_player(self, s: Tuple[int, int]) -> bool:
        """
        Whether a given square has a piece on it belonging to the player in turn.
        """
        return self._turn == np.sign(self._board[s])

    @staticmethod
    def square_is_outside_board(s: Tuple[int, int]) -> bool:
        """
        Whether a given square lies outside of the chessboard.
        """
        return True if max(s) > 7 or min(s) < 0 else False

    @staticmethod
    def new_board() -> np.ndarray:
        """
        Create a chessboard in starting position.
        """
        board = np.zeros((8, 8), dtype=np.int8)  # Initialize an all-zero 8x8 array
        board[1, :] = 1  # Set white pawns on row 2
        board[-2, :] = -1  # Set black pawns on row 7
        board[0, :] = [4, 2, 3, 5, 6, 3, 2, 4]  # Set white's main pieces on row 1
        board[-1, :] = -board[0]  # Set black's main pieces on row 8
        return board


class IllegalMoveError(Exception):
    pass
