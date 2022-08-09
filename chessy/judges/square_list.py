"""
Array-based judge and move-generator.
"""

# Standard library
from __future__ import annotations
from typing import Optional, NoReturn, Union, Tuple
from copy import deepcopy

# 3rd party
import numpy as np

# Self
from .abc import Judge, IllegalMoveError, GameOverError
from ..board_representation import BoardState, Move, Moves
from ..consts import *


class ArrayJudge(Judge):
    """

    Attributes
    ----------

    _board : numpy.ndarray

    _castling_rights : dict[int, numpy.ndarray]
        First element is a dummy element, so that `self._can_castle[self._turn]`
        gives the castling list of current player
    """

    def __init__(self, initial_state: BoardState):
        self._board: np.ndarray = initial_state.board.copy()
        self._castling_rights: dict = initial_state.castling_rights.copy()
        self._player: np.int8 = initial_state.player
        self._fifty_move_count: np.int8 = initial_state.fifty_move_count
        self._enpassant_file: np.int8 = initial_state.enpassant_file
        self._ply_count: np.int16 = initial_state.ply_count

        self._empty_array_squares = np.array([], dtype=np.int8).reshape(0, 2)
        self._empty_move: tuple = (self._empty_array_squares, self._empty_array_squares)

        self._is_checkmate: bool = False
        self._is_check: bool = False
        self._is_draw: bool = False
        self._valid_moves: Moves = None
        self.analyze_state()
        # If the BoardState is faulty, so that the opponent has been checkmated already in current
        # player's last move, now capturing opponent's king would be in current player's moves.
        opp_king_pos = self.squares_of_piece(p=self.opponent * KING)
        move_captures_king = np.all(self._valid_moves.s1s == opp_king_pos, axis=1)
        if np.any(move_captures_king):
            raise GameOverError(code=1)
        return

    @property
    def current_state(self) -> BoardState:
        """
        Current board-state as a `BoardState` object.
        """
        return BoardState(
            board=self._board.copy(),
            castling_rights=self._castling_rights.copy(),
            player=self.player,
            enpassant_file=self._enpassant_file,
            fifty_move_count=self._fifty_move_count,
            ply_count=self._ply_count,
        )

    @property
    def valid_moves(self) -> Moves:
        """
        All valid moves available to the current player in the current state.
        """
        return deepcopy(self._valid_moves)

    @property
    def is_checkmate(self) -> bool:
        """
        Whether the current player is checkmated in the current state.
        """
        return self._is_checkmate

    @property
    def is_draw(self) -> bool:
        """
        Whether the current state is a draw.
        """
        return self._is_draw

    def is_check(self) -> bool:
        """
        Whether the current player is in check.
        """
        return self._is_check

    @property
    def player(self) -> np.int8:
        """
        ID of the current player.
        """
        return self._player

    @property
    def opponent(self) -> np.int8:
        """
        ID of the current player's opponent.
        """
        return self.player * -1

    @property
    def occupied_squares(self) -> np.ndarray:
        """
        Coordinates of all non-empty squares on the board in the current state.
        """
        return np.argwhere(self._board != NULL)

    @property
    def squares_of_player(self):
        return np.argwhere(np.sign(self._board) == self.player)

    @property
    def move_is_promotion(self) -> bool:
        return

    @property
    def is_dead_position(self):
        return

    @property
    def pos_king(self):
        return self.squares_of_piece(self.king)[0]

    @property
    def king(self):
        return self.player * KING

    def submit_move(self, move: Move) -> NoReturn:
        move_vect = move.s1 - move.s0
        piece = self.pieces_in_squares(ss=move.s0)
        if self.is_checkmate:
            raise GameOverError(code=-1)
        if self.is_draw:
            raise GameOverError(code=0)
        if not self.squares_are_inside_board(ss=move.s0):
            raise IllegalMoveError(code=0)
        if not piece:
            raise IllegalMoveError(code=1)
        if not self.squares_belong_to_player(ss=move.s0):
            raise IllegalMoveError(code=2, player=self.player)
        if not self.squares_are_inside_board(ss=move.s1):
            raise IllegalMoveError(code=3)
        if np.all(move.s0 == move.s1):
            raise IllegalMoveError(code=4)
        if not self.move_principally_legal_for_piece(p=piece, move_vect=move_vect):
            raise IllegalMoveError(code=5, piece_type=self.piece_types(piece), move_vect=move_vect)
        if not self._valid_moves.has_move(move):
            if self.is_check:
                raise IllegalMoveError(code=6)
            raise IllegalMoveError(code=7)
        self.apply_move(move=move)
        return

    def apply_move(self, move: Move) -> None:
        piece_at_end_square = self.pieces_in_squares(ss=move.s0)
        moving_piece_type = self.piece_types(piece_at_end_square)
        captured_piece = self.pieces_in_squares(ss=move.s1)
        if captured_piece != NULL:
            self._fifty_move_count = -1
        move_vec = move.s1 - move.s0
        move_vec_mag = np.abs(move_vec)
        if moving_piece_type == PAWN:
            # Handle promotions and en passant
            self._fifty_move_count = -1
            if move.pp != NULL:
                piece_at_end_square = move.pp
            if np.all(move_vec_mag == [1, 1]) and captured_piece == 0:
                self._board[move.s1[0] - self.player, move.s1[1]] = 0
            self._enpassant_file = move.s1[1] if move_vec_mag[0] == 2 else -1
        else:
            self._enpassant_file = -1
            # Apply castling and/or modify castling rights
            if moving_piece_type == KING:
                self._castling_rights[self.player] = dict.fromkeys(
                    self._castling_rights[self.player], False
                )
                if move_vec_mag[1] == 2:
                    self._board[ROOK_S_INIT[self.player][move_vec[1]]] = 0
                    self._board[ROOK_S_END[self.player][move_vec[1]]] = self.player * ROOK
            elif moving_piece_type == ROOK:
                for side, pos in ROOK_S_INIT[self.player].items():
                    if np.all(move.s0 == pos):
                        self._castling_rights[self.player][side] = False
        self._board[tuple(move.s1)] = piece_at_end_square
        self._board[tuple(move.s0)] = 0
        self._fifty_move_count += 1
        self._ply_count += 1
        self._is_check = False
        self._player *= -1
        self.analyze_state()
        return

    def analyze_state(self):
        if self._fifty_move_count == 100 or self.is_dead_position:
            self._is_draw = True
        else:
            checking_squares = self.squares_leading_to()
            if checking_squares.size != 0:
                self._is_check = True
                valid_moves = self.generate_valid_moves_checked(ss_checking_king=checking_squares)
                if not valid_moves:
                    self._is_checkmate = True
            else:
                valid_moves = self.generate_valid_moves_unchecked()
                if not valid_moves:
                    self._is_draw = True
            self._valid_moves = valid_moves
        return

    def generate_valid_moves_unchecked(self) -> Moves:
        """
        Generate all the valid moves for the current player in the current state,
        knowing that the player is not in check.

        Returns
        -------
        list[Optional[Move]]
            A list of `Move` objects, or an empty list, if no valid move exists, in which case
            (knowing the player is not in check) it means the game is a draw due to stalemate.
        """
        s0s_p, s1s_p, pps = self.generate_pawn_moves()
        s0s_n, s1s_n = self.generate_knight_moves()
        s0s_b, s1s_b = self.generate_big_piece_moves(p=BISHOP)
        s0s_r, s1s_r = self.generate_big_piece_moves(p=ROOK)
        s0s_q, s1s_q = self.generate_big_piece_moves(p=QUEEN)
        s0s_k, s1s_k = self.generate_king_moves()
        s0s = [s0s_p, s0s_n, s0s_b, s0s_r, s0s_q, s0s_k]
        s1s = [s1s_p, s1s_n, s1s_b, s1s_r, s1s_q, s1s_k]
        move_counts = [s.shape[0] for s in s0s]
        ps = np.repeat(
            np.array([PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING], dtype=np.int8), move_counts
        )
        pps = np.concatenate([pps, np.zeros(sum(move_counts[1:]), dtype=np.int8)])
        return Moves(s0s=np.concatenate(s0s), s1s=np.concatenate(s1s), ps=ps, pps=pps)

    def generate_valid_moves_checked(self, ss_checking_king: np.ndarray) -> Moves:
        # Initialize empty list to accumulate different moves
        s0s_all, s1s_all, ps_all, pps_all = [], [], [], []
        # Get king moves and add to lists
        s0s_k, s1s_k = self.generate_king_moves()
        ps_k = np.ones(shape=s0s_k.shape[0], dtype=np.int8) * self.player * KING
        pps_k = np.zeros(shape=s0s_k.shape[0], dtype=np.int8)
        s0s_all.append(s0s_k)
        s1s_all.append(s1s_k)
        ps_all.append(ps_k)
        pps_all.append(pps_k)
        # In case of single checks, also get the moves that capture or block the checking piece.
        if ss_checking_king.shape[0] == 1:
            s0s_c, s1s_c, ps_c, pps_c = self.generate_targeted_moves(
                s1=ss_checking_king[0], mode="attacking"
            )
            s0s_all.append(s0s_c)
            s1s_all.append(s1s_c)
            ps_all.append(ps_c)
            pps_all.append(pps_c)
            # Get the square in between the checking piece and king
            for s_between_king_checker in self.squares_in_between(
                s0=ss_checking_king[0], s1=self.pos_king
            ):
                s0s_a, s1s_a, ps_a, pps_a = self.generate_targeted_moves(
                    s1=s_between_king_checker, mode="advancing"
                )
                s0s_all.append(s0s_a)
                s1s_all.append(s1s_a)
                ps_all.append(ps_a)
                pps_all.append(pps_a)
        return Moves(
            s0s=np.concatenate(s0s_all),
            s1s=np.concatenate(s1s_all),
            ps=np.concatenate(ps_all),
            pps=np.concatenate(pps_all)
        )

    def generate_targeted_moves(self, s1: np.ndarray, mode: str):
        # Get the squares that can capture the checking piece,
        s0s = self.squares_leading_to(s=s1, p=self.player, status=mode)
        ps = self.pieces_in_squares(ss=s0s)
        # If the end-square is not on the promotion rank of current player,
        # then no promotion is possible
        if not self.mask_ss_non_p_rank(ss=s1):
            pps = np.zeros(s0s.shape[0], dtype=np.int8)
        else:
            # Otherwise, check for possible promotions; all capturing pawns will be promoted
            mask_nopromo_s0s = ps == self.player
            # Create repetition count array to generate promotion moves
            reps_for_promotion, pps = self.create_promotion_data(mask_no_promo=mask_nopromo_s0s)
            s0s = np.repeat(s0s, reps_for_promotion, axis=0)
            ps = np.repeat(ps, reps_for_promotion)
        return s0s, np.tile(s1, reps=s0s.shape[0]).reshape(-1, 2), ps, pps

    def generate_big_piece_moves(self, p: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate valid moves for queens, rooks and bishops
        of the current player in the current state.

        Returns
        -------
        list[Optional[Move]]
            A list of `Move` objects, or an empty list, if no valid move exists.
        """
        s0s = self.squares_of_piece(p=p * self.player)  # Get the start-squares.
        if s0s.size == 0:  # If no piece of that kind is on board, continue to next piece.
            return self._empty_move
        # For each start-square, go one square in every direction that piece can move in.
        s1s = s0s[:, np.newaxis] + MOVE_DIRECTIONS[p]
        # Get the end squares that are still in-board, to eliminate searching in
        # directions leaving the board (when piece is on one or two edges).
        mask_inboard = self.squares_are_inside_board(ss=s1s)
        s1s_valid = s1s[mask_inboard]
        # Repeat each start square as many times as the number of remaining directions from
        # that square, to get an array of start-squares corresponding one-to-one with the
        # array of end-squares.
        s0s_valid = np.repeat(s0s, np.count_nonzero(mask_inboard, axis=1), axis=0)
        if s1s_valid.size == 0:  # If no direction is remained, go to next piece.
            return self._empty_move
        # From the remaining end-squares, get those that are vacant
        # (i.e. empty or occupied by opponent), to avoid searching in dead-end directions.
        mask_vacant = ~self.squares_belong_to_player(ss=s1s_valid)
        if not np.any(mask_vacant):  # If no direction is remained, go to next piece.
            return self._empty_move
        # From the remaining end-squares, get those in directions
        # that don't break an absolute pin.
        mask_unpinned = self.mask_absolute_pin(
            s0s=s0s_valid[mask_vacant], s1s=s1s_valid[mask_vacant]
        )
        if not np.any(mask_unpinned):  # If no direction is remained, go to next piece.
            return self._empty_move
        # Filter the arrays of start-squares and end-squares, based on above masks
        s0s_valid = s0s_valid[mask_vacant][mask_unpinned]
        s1s_valid = s1s_valid[mask_vacant][mask_unpinned]
        # Calculate back the remaining valid directions
        ds_valid = s1s_valid - s0s_valid
        # Get the coordinates of all neighbors of all remaining start-squares
        neighbors_pos = self.neighbor_squares(ss=s0s_valid, ds=ds_valid)
        # Get the pieces on those squares
        neighbor_pieces = self.pieces_in_squares(neighbors_pos)
        # Create a shift array (elements 0 or 1) to account for neighbors that belong to the
        # player (and thus cannot be moved into), vs. neighbors that are empty or belong to
        # the opponent (and thus can be moved into).
        shift_for_own_piece = self.pieces_belong_to_player(ps=neighbor_pieces)
        # Calculate the maximum allowed magnitude of move for every direction, based on the
        # coordinates of the neighbor in that direction, and using the shift array.
        move_mag_restricted = np.abs(neighbors_pos - s0s_valid).max(axis=-1) - shift_for_own_piece
        # Get magnitudes greater than zero (i.e. a move in that direction is possible)
        mask_valid_moves = move_mag_restricted > 0
        if not np.any(mask_valid_moves):  # If no direction is remained, go to next piece.
            return self._empty_move
        valid_move_mags = move_mag_restricted[mask_valid_moves]
        valid_move_dirs = ds_valid[mask_valid_moves]
        # For each direction, generate as many possible move-vectors in that direction, e.g.
        # if move direction is (1,1) and max. magnitude is 3, generate [(1, 1), (2, 2), (3, 3)]
        valid_moves_ = [
            d * np.expand_dims(np.arange(1, mag + 1, dtype=np.int8), 1)
            for mag, d in zip(valid_move_mags, valid_move_dirs)
        ]
        valid_moves_ = np.concatenate(valid_moves_, dtype=np.int8)
        # Again, repeat each remaining start-square as many times as the max. magnitude of the
        # move in that direction.
        s0s_valid = np.repeat(s0s_valid[mask_valid_moves], valid_move_mags, axis=0)
        # Add each start-square to its corresponding move-vector to get the actual end-squares.
        s1s_valid = (s0s_valid + valid_moves_).reshape(-1, 2)
        return s0s_valid, s1s_valid

    def generate_knight_moves(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate valid moves for knights of the current player in the current state.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Start- and end-squares of all valid moves.
        """
        # The procedure is similar to the one for big pieces, but here we don't have to check for
        # neighbors and calculate move magnitudes, since the knight can jump.
        s0s = self.squares_of_piece(p=self.player * KNIGHT)
        if s0s.size == 0:
            return self._empty_move
        s1s = s0s[:, np.newaxis] + MOVE_DIRECTIONS[KNIGHT]
        mask_inboard = self.squares_are_inside_board(ss=s1s)
        s0s_valid = np.repeat(s0s, np.count_nonzero(mask_inboard, axis=1), axis=0)
        s1s_valid = s1s[mask_inboard]
        mask_vacant = ~self.squares_belong_to_player(ss=s1s_valid)
        if not np.any(mask_vacant):
            return self._empty_move
        mask_unpinned = self.mask_absolute_pin(
            s0s=s0s_valid[mask_vacant], s1s=s1s_valid[mask_vacant]
        )
        return s0s_valid[mask_vacant][mask_unpinned], s1s_valid[mask_vacant][mask_unpinned]

    def generate_pawn_moves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: FIX ENPASSANTS LEADING TO CHECK ALONG A RANK
        s0s = self.squares_of_piece(p=self.player)
        if s0s.size == 0:
            return self._empty_move
        s1s_all = s0s[:, np.newaxis] + MOVE_DIRECTIONS[PAWN] * self.player
        s1s_forward, s1s_attack = s1s_all[:, :2], s1s_all[:, 2:]
        s1s_single, s1s_double = s1s_forward[:, 0], s1s_forward[:, 1]
        running_mask = self.squares_are_inside_board(ss=s1s_all)
        mask_vacant_forward1 = self.squares_are_empty(ss=s1s_single[running_mask[:, 0]])
        mask_vacant_forward2 = self.squares_are_empty(ss=s1s_double[running_mask[:, 1]])
        running_mask[:, 0][running_mask[:, 0]] &= mask_vacant_forward1
        running_mask[:, 1][running_mask[:, 1]] &= mask_vacant_forward2
        if np.any(running_mask[:, 0]):
            unpin_mask_forward = self.mask_absolute_pin(
                s0s=s0s[running_mask[:, 0]], s1s=s1s_single[running_mask[:, 0]]
            )
            running_mask[:, 0][running_mask[:, 0]] &= unpin_mask_forward
            running_mask[:, 1] &= running_mask[:, 0]
            mask_in_initial_pos = self.mask_ss_non_p_rank(ss=s0s)
            running_mask[:, 1] &= mask_in_initial_pos
        mask_can_attack = self.squares_belong_to_opponent(
            ss=s1s_attack[running_mask[:, 2:]]
        ) | self.is_enpassant_square(ss=s1s_attack[running_mask[:, 2:]])
        running_mask[:, 2:][running_mask[:, 2:]] &= mask_can_attack
        if np.any(running_mask[:, 2]):
            unpin_mask_attack1 = self.mask_absolute_pin(
                s0s=s0s[running_mask[:, 2]], s1s=s1s_attack[:, 0][running_mask[:, 2]]
            )
            running_mask[:, 2][running_mask[:, 2]] &= unpin_mask_attack1
        if np.any(running_mask[:, 3]):
            unpin_mask_attack2 = self.mask_absolute_pin(
                s0s=s0s[running_mask[:, 3]], s1s=s1s_attack[:, 1][running_mask[:, 3]]
            )
            running_mask[:, 3][running_mask[:, 3]] &= unpin_mask_attack2
        if not np.any(running_mask):
            return self._empty_move

        s0s = np.repeat(s0s, np.count_nonzero(running_mask, axis=1), axis=0)
        s1s = s1s_all[running_mask]

        mask_no_promotion = self.mask_ss_non_prom_rank(ss=s1s)
        reps_for_promotion, pps = self.create_promotion_data(mask_no_promo=mask_no_promotion)
        s0s = np.repeat(s0s, reps_for_promotion, axis=0)
        s1s = np.repeat(s1s, reps_for_promotion, axis=0)
        return s0s, s1s, pps

    def generate_king_moves(self) -> tuple[np.ndarray, np.ndarray]:
        s1s_all = self.pos_king + MOVE_DIRECTIONS[KING]
        s1s_normal, s1s_castle = s1s_all[:-2], s1s_all[-2:]
        s1s_inboard = s1s_normal[ArrayJudge.squares_are_inside_board(ss=s1s_normal)]
        s1s_vacant = s1s_inboard[~self.squares_belong_to_player(ss=s1s_inboard)]
        s1s_final = s1s_vacant[self.king_wont_be_attacked(ss=s1s_vacant)]
        player_castling_rights = np.array(list(self._castling_rights[self.player].values()))
        if not self.is_check and np.any(player_castling_rights):
            vacant = self.squares_are_empty(ss=CASTLING_SS_EMPTY[self.player])
            mask_vacant = [np.all(vacant[:3]), np.all(vacant[3:])]
            not_checked = self.king_wont_be_attacked(ss=CASTLING_SS_CHECK[self.player])
            mask_not_checked = np.all(not_checked.reshape(2, 2), axis=1)
            mask_castle = mask_vacant & mask_not_checked & player_castling_rights
            s1s_final_castle = s1s_castle[mask_castle]
            s1s_final = np.concatenate((s1s_final.reshape(-1, 2), s1s_final_castle.reshape(-1, 2)))
        return np.tile(self.pos_king, reps=s1s_final.shape[0]).reshape(-1, 2), s1s_final

    def king_wont_be_attacked(self, ss: np.ndarray):
        king_pos = tuple(self.pos_king)
        self._board[king_pos] = 0  # temporarily remove king from board
        square_is_not_attacked = []
        for square in ss:
            piece_in_square = self.pieces_in_squares(ss=square)
            self._board[tuple(square)] = 0  # temporarily remove the piece from the square
            square_is_not_attacked.append(self.squares_leading_to(s=square).size == 0)
            self._board[tuple(square)] = piece_in_square  # put the piece back
        self._board[king_pos] = self.king  # put the king back
        return np.array(square_is_not_attacked, dtype=np.bool_)

    def squares_leading_to(
        self, s: Optional[np.ndarray] = None, p: Optional[np.int8] = None, status: str = "checking"
    ) -> np.ndarray:
        """
        Coordinates of squares containing a given player's pieces that currently
        attack a given square. This is regardless of whether the player's pieces on those
        squares are under absolute pin or not. Thus, this method returns the squares that
        currently 'check' a given square.

        Parameters
        ----------
        s : Optional[numpy.ndarray(shape=(2, ), dtype=numpy.int8)]
            Coordinates of the square to be checked.
            Defaults to the king square of the player that is not `p`.
        p : Optional[numpy.int8]
            Player whose pieces are attacking.
            Defaults to the opponent of the current player.
        Returns
        -------
        numpy.ndarray[(shape=(n, 2), dtype=numpy.int8)]
            Coordinates of the 'checking' squares.
        """
        p = self.opponent if p is None else p
        s = self.squares_of_piece(-p * KING)[0] if s is None else s
        # 1. CHECK FOR KNIGHT ATTACKS
        # Add given start-square to all knight vectors to get all possible attacking positions
        knight_pos = s + MOVE_DIRECTIONS[KNIGHT]
        # Take those end squares that are within the board
        inboards = knight_pos[self.squares_are_inside_board(ss=knight_pos)]
        mask_knight = self.pieces_in_squares(inboards) == p * KNIGHT
        # 2. CHECK FOR STRAIGHT-LINE ATTACKS (queen, bishop, rook, pawn, king)
        # Get nearest neighbor in each direction
        neighbors_pos = self.neighbor_squares(
            ss=np.tile(s, 8).reshape(-1, 2), ds=DIRECTIONS * p
        )
        neighbors = self.pieces_in_squares(neighbors_pos)
        # For queen, rook and bishop, if they are in neighbors, then it means they are attacking
        # mask_king = (neighbors == opp_pieces[6]) & (np.abs(neighbors_pos - s).max(axis=1) == 1)
        mask_queen = neighbors == p * QUEEN
        mask_rook = neighbors[::2] == p * ROOK
        mask_bishop = neighbors[1::2] == p * BISHOP

        if status != "advancing":
            pawn_dirs = [3, 5]
            mask_pawn_right_distance = (neighbors_pos[pawn_dirs] - s)[:, 0] == -p
        else:
            pawn_dirs = 4
            dir_mag_vertical = (neighbors_pos[pawn_dirs] - s)[0]
            mask_pawn_right_distance = (dir_mag_vertical == -p) | (
                (dir_mag_vertical == -p * 2)
                & (neighbors_pos[pawn_dirs, 0] == (0 if p == 1 else 6))
            )
        mask_pawn_right_direction = neighbors[pawn_dirs] == p * PAWN
        mask_pawn = mask_pawn_right_direction & mask_pawn_right_distance

        leading_squares = np.concatenate(
            [
                inboards[mask_knight],
                # neighbors_pos[mask_king],
                neighbors_pos[mask_queen],
                neighbors_pos[::2][mask_rook],
                neighbors_pos[1::2][mask_bishop],
                neighbors_pos[pawn_dirs][mask_pawn],
            ]
        )

        if status != "checking":
            mask_absolute_pin = self.mask_absolute_pin(
                s0s=leading_squares, s1s=np.tile(s, leading_squares.size // 2).reshape(-1, 2)
            )
            leading_squares = leading_squares[mask_absolute_pin]

        return leading_squares

    def mask_absolute_pin(self, s0s, s1s):
        current_unpin_mask = np.zeros(s0s.size // 2, dtype=np.bool_)

        _, s0ks_uv, _, s0ks_cardinal_mask = ArrayJudge.move_dir_mag(s0s=s0s, s1s=self.pos_king)
        current_unpin_mask[~s0ks_cardinal_mask] = True

        _, s01s_uv, _, _ = ArrayJudge.move_dir_mag(s0s=s0s, s1s=s1s)
        move_towards_king = s01s_uv[~current_unpin_mask] == s0ks_uv[~current_unpin_mask]
        move_opposite_king = s01s_uv[~current_unpin_mask] == -s0ks_uv[~current_unpin_mask]
        mask_move_along_s0k = (move_towards_king[..., 0] & move_towards_king[..., 1]) | (
            move_opposite_king[..., 0] & move_opposite_king[..., 1]
        )
        current_unpin_mask[~current_unpin_mask] = mask_move_along_s0k

        kingside_neighbors_squares = self.neighbor_squares(
            ss=s0s[~current_unpin_mask],
            ds=s0ks_uv[~current_unpin_mask],
        )
        kingside_neighbors = self.pieces_in_squares(ss=kingside_neighbors_squares)
        mask_king_protected = kingside_neighbors != self.king
        current_unpin_mask[~current_unpin_mask] = mask_king_protected

        otherside_neighbors_squares = self.neighbor_squares(
            ss=s0s[~current_unpin_mask],
            ds=-s0ks_uv[~current_unpin_mask],
        )
        otherside_neighbors = self.pieces_in_squares(ss=otherside_neighbors_squares)
        no_queen = otherside_neighbors != self.opponent * QUEEN
        has_orthogonal_dir = s0ks_uv[~current_unpin_mask] == 0
        is_orthogonal = has_orthogonal_dir[..., 0] | has_orthogonal_dir[..., 1]
        no_rooks = otherside_neighbors[is_orthogonal] != self.opponent * ROOK
        no_bishops = otherside_neighbors[~is_orthogonal] != self.opponent * BISHOP
        mask_no_pinning = no_queen
        mask_no_pinning[is_orthogonal] &= no_rooks
        mask_no_pinning[~is_orthogonal] &= no_bishops
        current_unpin_mask[~current_unpin_mask] = mask_no_pinning
        return current_unpin_mask

    def neighbor_squares(self, ss, ds):
        neighbor_squares = np.zeros(shape=ss.shape, dtype=np.int8)
        next_neighbors_pos = ss + ds
        not_set = np.ones(ss.size // 2, dtype=np.bool_)
        while np.any(not_set):
            mask_inboard = self.squares_are_inside_board(ss=next_neighbors_pos[not_set])
            curr_mask = not_set.copy()
            curr_mask[curr_mask] = ~mask_inboard
            neighbor_squares[curr_mask] = next_neighbors_pos[curr_mask] - ds[curr_mask]
            not_set[not_set] = mask_inboard
            inboard_squares = next_neighbors_pos[not_set]
            mask_occupied = self.pieces_in_squares(ss=inboard_squares) != 0
            curr_mask = not_set.copy()
            curr_mask[curr_mask] = mask_occupied
            neighbor_squares[curr_mask] = inboard_squares[mask_occupied]
            not_set[not_set] = ~mask_occupied
            next_neighbors_pos += ds
        return neighbor_squares

    def pawn_move_restriction(self, s0):
        move_dirs = np.array([[1, 0], [1, 1], [1, -1]], dtype=np.int8) * self.player
        return np.array(
            [
                1 + self.mask_ss_non_p_rank(s0),
                self.pawn_can_capture_square(s0 + move_dirs[1]),
                self.pawn_can_capture_square(s0 + move_dirs[2]),
            ],
            dtype=np.int8,
        )

    def is_enpassant_square(self, ss):
        return np.all(ss == [RANK_ENPASSANT_END[self.player], self._enpassant_file], axis=-1)

    def pawn_can_capture_square(self, s1):
        can_capture_enpassant = np.all(
            s1 == [RANK_ENPASSANT_END[self.player], self._enpassant_file]
        )
        can_capture_normal = self.squares_are_inside_board(
            ss=s1
        ) and self.squares_belong_to_opponent(ss=s1)
        return can_capture_normal or can_capture_enpassant

    def castling_right(self, side: int) -> bool:
        """
        Whether current player has castling right for the given side.

        Parameters
        ----------
        side : int
            +1 for kingside, -1 for queenside.
        """
        return self._castling_rights[self.player][side]

    def pieces_in_squares(self, ss: np.ndarray) -> Union[np.ndarray, np.int8]:
        """
        Get the type of pieces on a given number of squares.

        Parameters
        ----------
        ss : numpy.ndarray
          Coordinates of n squares as an array of x dimensions
          with shape (s_1, s_2, ..., s_{x-1}, 2), where the last dimension
          corresponds to the file/rank coordinates. For the rest of the dimensions,
          it holds that:  s_1 * s_2 * ... * s_{x-1} = n

        Returns
        -------
        Union[np.ndarray, np.int8]
            Piece types as a single integer (when `ss` is 1-dimensional) or a 1d-array of size n.
        """
        if ss.size > 1:
            return self._board[ss[..., 0], ss[..., 1]]
        else:
            return np.array([], dtype=np.int8)

    def squares_of_piece(self, p: int):
        return np.argwhere(self._board == p)

    def squares_belong_to_player(self, ss: np.ndarray) -> bool:
        """
        Whether a given square has a piece on it belonging to the player in turn.
        """
        return self.squares_belong_to(ss=ss, p=self.player)

    def squares_belong_to_opponent(self, ss: np.ndarray) -> bool:
        """
        Whether a given square has a piece on it belonging to the opponent.
        """
        return self.squares_belong_to(ss=ss, p=self.opponent)

    def squares_belong_to(self, ss: np.ndarray, p: np.int8) -> bool:
        """
        Whether a given square has a piece on it belonging to a given player.
        """
        return np.sign(self.pieces_in_squares(ss=ss)) == p

    def squares_are_empty(self, ss: np.ndarray):
        return self.pieces_in_squares(ss=ss) == 0

    def pieces_belong_to_player(
        self, ps: Union[np.int8, np.ndarray]
    ) -> Union[np.bool_, np.ndarray]:
        return np.sign(ps) == self.player

    def pieces_belong_to_opponent(
        self, ps: Union[np.int8, np.ndarray]
    ) -> Union[np.bool_, np.ndarray]:
        return np.sign(ps) == self.opponent

    def mask_ss_non_p_rank(self, ss: np.ndarray) -> np.ndarray:
        """
        Get a boolean mask array to select squares residing on the pawn-rank
        of the current player (i.e. rank 2 for white, rank 7 for black).

        Parameters
        ----------
        ss : numpy.ndarray[shape=(n, 2)]
            Coordinates of the squares to be tested.

        Returns
        -------
        numpy.ndarray[shape=(n, ), dtype=numpy.bool_]
            A boolean array that can be used as a mask for the input array
            (i.e. `ss[mask_ss_non_p_rank(ss)]`) to obtain the squares that are
            on the pawn-rank of the current player. This can be used, e.g. to check which
            pawns of the current player have not been moved yet.
        """
        return ss[..., 0] == RANK_PAWN[self.player]

    def mask_ss_non_prom_rank(self, ss: np.ndarray) -> np.ndarray:
        """
        Get a boolean mask array to select squares residing on the promotion-rank
        of the current player (i.e. rank 8 for white, rank 0 for black).

        Parameters
        ----------
        ss : numpy.ndarray[shape=(n, 2)]
            Coordinates of the squares to be tested.

        Returns
        -------
        numpy.ndarray[shape=(n, ), dtype=numpy.bool_]
            A boolean array that can be used as a mask for the input array
            (i.e. `ss[mask_ss_non_p_rank(ss)]`) to obtain the squares that are
            on the promotion-rank of the current player. This can be used, e.g. to find moves
            leading to promotion.
        """
        return ss[..., 0] == RANK_PROMOTION[self.player]

    def squares_in_between(self, s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
        """
        Indices of the squares between two given squares
        lying on an orthogonal/diagonal direction.

        Parameters
        ----------
        s0 : numpy.ndarray(shape=(2,), dtype=np.int8)
        s1 : numpy.ndarray(shape=(2,), dtype=np.int8)

        Returns
        -------
        numpy.ndarray(shape=(n, 2))
            The indices of n squares starting from s0 and going to s1.
            If the squares are the same, or not on an orthogonal/diagonal direction, then
            an empty array of shape (0, 2) is returned.
        """
        move_v, move_uv, move_vm, is_cardinal = ArrayJudge.move_dir_mag(s0s=s0, s1s=s1)
        if not is_cardinal:
            return self._empty_array_squares
        return s0 + np.arange(1, move_vm, dtype=np.int8)[:, np.newaxis] * move_uv

    @staticmethod
    def create_promotion_data(mask_no_promo: np.ndarray):
        # Get 4 for promoting moves, and 1 for non-promoting moves
        reps_for_promotion = mask_no_promo * 3 + 1
        pps = []  # Create promotion array
        for is_promotion in mask_no_promo:
            pps.extend([KNIGHT, BISHOP, ROOK, QUEEN] if is_promotion else [NULL])
        pps = np.array(pps, dtype=np.int8)
        return reps_for_promotion, pps

    @staticmethod
    def move_dir_mag(s0s: np.ndarray, s1s: np.ndarray) -> tuple:
        """
        Vector properties for between a number of start and end-squares.

        Parameters
        ----------
        s0s : numpy.ndarray
            Coordinates of n start-squares as an array of x dimensions
            with shape (s_1, s_2, ..., s_{x-1}, 2), where the last dimension
            corresponds to the file/rank coordinates. For the rest of the dimensions,
            it holds that:  s_1 * s_2 * ... * s_{x-1} = n
        s1s : numpy.ndarray
            Coordinates of n end-squares as an array with same shape as `s0s`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, np.int8], Union[np.ndarray, bool]]
            [0]: Vectors from s0s to s1s, as an array with same shape as s0s/s1s.
            [1]: Corresponding smallest-int (unit) vectors, as an array with same shape as s0s/s1s.
            [2]: Unit vector multipliers (i.e. the numbers that multiplied with corresponding
                 unit vectors give the original vectors) as an array with shape
                 (s_1, s_2, ..., s_{x-1}), or an integer when `s0s` and `s1s` are 1-dimensional.
            [3]: Whether the vectors are cardinal (i.e. non-zero and along a cardinal direction),
                 as a boolean array with shape (s_1, s_2, ..., s_{x-1}), or a single boolean when
                 `s0s` and `s1s` are 1-dimensional.
        """
        move_vect = s1s - s0s
        move_vect_multiplier = np.gcd(move_vect[..., 0], move_vect[..., 1])
        move_unit_vect = move_vect // np.array(move_vect_multiplier)[..., np.newaxis]
        is_cardinal = np.abs(move_unit_vect).max(axis=-1) == 1
        return move_vect, move_unit_vect, move_vect_multiplier, is_cardinal

    @staticmethod
    def squares_are_inside_board(ss: np.ndarray) -> np.ndarray:
        """
        Whether a number of given squares lie inside the board.

        Parameters
        ----------
        ss : numpy.ndarray
          Coordinates of n squares as an array of x dimensions
          with shape (s_1, s_2, ..., s_{x-1}, 2), where the last dimension
          corresponds to the file/rank coordinates. For the rest of the dimensions,
          it holds that:  s_1 * s_2 * ... * s_{x-1} = n

        Returns
        -------
        numpy.ndarray
          A boolean value (when ss.shape=(2,)) or array with shape (s_1, s_2, ..., s_{x-1}).
        """
        # if ss.ndim == 1:
        #     ss = np.expand_dims(ss, axis=0)
        # return np.all(np.all([ss < 8, ss > -1], axis=0), axis=1)
        rank_file_inside_board = (ss > -1) & (ss < 8)
        return rank_file_inside_board[..., 0] & rank_file_inside_board[..., 1]

    @staticmethod
    def piece_types(ps: Union[np.int8, np.ndarray]) -> Union[np.int8, np.ndarray]:
        """
        Piece-type of a number of given pieces.
        Piece-type is the absolute value of a piece data, as defined in `BoardState`, i.e.
        0 = empty, 1 = pawn, 2 = knight, 3 = bishop, 4 = rook, 5 = queen, 6 = king.

        Parameters
        ----------
        ps : Union[np.int8, np.ndarray]
            Piece data as defined by `BoardState`.
        Returns
        -------
        Union[np.int8, np.ndarray]
            Piece-type, either as a single integer, or an array of integers, depending on input.
        """
        return np.abs(ps)

    @staticmethod
    def move_principally_legal_for_piece(p: np.int8, move_vect: np.ndarray) -> bool:
        move_abs = np.abs(move_vect)
        move_manhattan_dist = move_abs.sum()
        piece_type = abs(p)
        if piece_type == PAWN:
            return (move_vect[0] == p and move_abs[1] < 2) or np.all(move_vect == [2 * p, 0])
        elif piece_type == KNIGHT:
            return not (move_manhattan_dist != 3 or np.isin(3, move_abs))
        elif piece_type == BISHOP:
            return move_abs[0] == move_abs[1]
        elif piece_type == ROOK:
            return np.isin(0, move_abs)
        elif piece_type == QUEEN:
            return move_abs[0] == move_abs[1] or np.isin(0, move_abs)
        elif piece_type == KING:
            return move_manhattan_dist == 1 or (move_manhattan_dist == 2 and move_abs[0] != 2)
