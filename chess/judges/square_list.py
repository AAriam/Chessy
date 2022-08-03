from typing import Optional, Sequence, NamedTuple, Any, NoReturn, Union, Tuple

import numpy as np

from .abc import Judge, IllegalMoveError, GameOverError
from ..board_representation import BoardState, Move, COLOR, PIECE


def move_vectors_orthogonal():
    moves = np.zeros((28, 2), dtype=np.int8)
    mag = np.arange(1, 8)
    moves[:7, 0] = mag
    moves[7:14, 0] = -mag
    moves[14:21, 1] = mag
    moves[21:, 1] = -mag
    return moves


def move_vectors_diagonal():
    moves = np.zeros((28, 2), dtype=np.int8)
    mag = np.repeat(np.arange(1, 8), 2).reshape(-1, 2)
    moves[:7] = mag
    moves[7:14] = -mag
    moves[14:21] = mag * [1, -1]
    moves[21:] = mag * [-1, 1]
    return moves


def move_vectors_cardinal():
    return np.concatenate([move_vectors_orthogonal(), move_vectors_diagonal()])


def move_vectors_cardinal_unit():
    return np.array(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.int8
    )


class ArrayJudge(Judge):
    """

    Attributes
    ----------

    board : numpy.ndarray

    castling_rights : numpy.ndarray
        First element is a dummy element, so that `self._can_castle[self._turn]`
        gives the castling list of current player
    """

    # Directions: top, bottom, right, left, top-right, top-left, bottom-right, bottom-left
    DIRECTION_UNIT_VECTORS = move_vectors_cardinal_unit()

    UNIT_VECTORS_ORTHO = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int8)
    UNIT_VECTORS_DIAG = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int8)

    # All possible move vectors for a piece
    MOVE_VECTORS = {
        1: np.array([[1, 0], [1, 1], [1, -1]], dtype=np.int8),
        2: np.array(
            [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]], dtype=np.int8
        ),
        3: np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int8),
        4: np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int8),
        5: np.array(
            [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.int8
        ),
        6: move_vectors_cardinal_unit(),
    }

    MOVE_VECTORS_PIECE = {
        11: np.array([[1, 0]], dtype=np.int8),  # Pawn vertical advance
        12: np.array([[2, 0]], dtype=np.int8),  # Pawm double vertical advance
        13: np.array([[1, 1], [1, -1]], dtype=np.int8),  # Pawn attacks
        2: np.array(  # Knight moves
            [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]], dtype=np.int8
        ),
        3: UNIT_VECTORS_DIAG,
        4: UNIT_VECTORS_ORTHO,
        5: np.concatenate([UNIT_VECTORS_ORTHO, UNIT_VECTORS_DIAG]),
        61: DIRECTION_UNIT_VECTORS,  # King normal moves
        62: np.array([[0, -2], [0, 2]], dtype=np.int8),  # King castling moves
    }

    def __init__(
        self,
        board: np.ndarray,
        castling_rights: np.ndarray,
        player: np.int8,
        enpassant_file: np.int8,
        fifty_move_count: np.int8,
        ply_count: np.int16,
    ):
        self.board: np.ndarray = board
        self.castling_rights: np.ndarray = castling_rights
        self.player: np.int8 = player
        self.fifty_move_count: np.int8 = fifty_move_count
        self.enpassant_file: np.int8 = enpassant_file
        self.ply_count: np.int16 = ply_count

        self.is_checkmate: bool = False
        self.is_check: bool = False
        self.is_draw: bool = False
        self.valid_moves: list[Move] = []
        self.analyze_state()
        return

    @classmethod
    def load_state(cls, state: BoardState) -> Judge:
        return cls(
            board=state.board,
            castling_rights=np.pad(state.castling_rights, (1, 0)),
            player=state.player,
            enpassant_file=state.enpassant_file,
            fifty_move_count=state.fifty_move_count,
            ply_count=state.ply_count,
        )

    def reveal_current_state(self) -> BoardState:
        return BoardState(
            board=self.board,
            castling_rights=self.castling_rights[1:, 1:],
            player=self.player,
            enpassant_file=self.enpassant_file,
            fifty_move_count=self.fifty_move_count,
            ply_count=self.ply_count,
        )

    def submit_move(self, move: Move) -> NoReturn:
        move_vect = move.end_square - move.start_square
        piece = self.piece_in_squares(ss=move.start_square)
        piece_name = PIECE[self.piece_types(piece)].name
        if self.is_checkmate:
            raise GameOverError("Game over. Current player is checkmated.")
        if self.is_draw:
            raise GameOverError("Game over. It is a draw.")
        if not self.squares_are_inside_board(ss=move.start_square):
            raise IllegalMoveError("Start-square is out of board.")
        if not piece:
            raise IllegalMoveError("Start-square is empty.")
        if not self.squares_belong_to_player(ss=move.start_square):
            raise IllegalMoveError(f"It is {COLOR[self.player].name}'s turn.")
        if not self.squares_are_inside_board(ss=move.end_square):
            raise IllegalMoveError("End-square is out of board.")
        if np.all(move.start_square == move.end_square):
            raise IllegalMoveError("Start and end-square are the same.")
        if not self.move_principally_legal_for_piece(p=piece, move_vect=move_vect):
            raise IllegalMoveError(
                f"{piece_name.capitalize()}s cannot move in direction {move_vect}."
            )
        if move not in self.valid_moves:
            if self.is_check:
                raise IllegalMoveError("Move does not resolve check.")
            raise IllegalMoveError("Submitted move is illegal.")
        self.apply_move(move=move)
        return

    def apply_move(self, move: Move) -> None:
        piece_at_end_square = self.piece_in_squares(ss=move.start_square)
        moving_piece_type = self.piece_types(piece_at_end_square)
        captured_piece = self.piece_in_squares(ss=move.end_square)
        if captured_piece != 0:
            self.fifty_move_count = -1
        move_vec = move.end_square - move.start_square
        move_vec_mag = np.abs(move_vec)
        if moving_piece_type == 1:
            # Handle promotions and en passant
            self.fifty_move_count = -1
            if move.promote_to is not None:
                piece_at_end_square = move.promote_to * self.player
            if np.all(move_vec_mag == [1, 1]) and captured_piece == 0:
                self.board[move.end_square[0] - self.player, move.end_square[1]] = 0
            self.enpassant_file = move.end_square[1] if move_vec_mag[0] == 2 else -1
        else:
            self.enpassant_file = -1
            # Apply castling and/or modify castling rights
            if moving_piece_type == 6:
                self.castling_rights[self.player] = 0
                if move_vec_mag[1] == 2:
                    rook_pos = (move.end_square[0], 7 if move_vec[1] == 2 else 0)
                    rook_end_pos = (move.end_square[0], 5 if move_vec[1] == 2 else 3)
                    self.board[rook_pos] = 0
                    self.board[rook_end_pos] = 4 * self.player
            elif moving_piece_type == 4:
                if move.start_square[1] == 0:
                    self.castling_rights[self.player, -1] = 0
                elif move.start_square[1] == 7:
                    self.castling_rights[self.player, 1] = 0
        self.board[tuple(move.end_square)] = piece_at_end_square
        self.board[tuple(move.start_square)] = 0
        self.fifty_move_count += 1
        self.player *= -1
        self.analyze_state()
        return

    def generate_all_valid_moves(self) -> list[Move]:
        return self.valid_moves

    def analyze_state(self):
        if self.fifty_move_count == 100 or self.is_dead_position:
            self.is_draw = True
            self.valid_moves = []
        else:
            cheking_squares = self.squares_checking()
            if cheking_squares.size != 0:
                self.is_check = True
                valid_moves = self.moves_resolving_check(attacking_squares=cheking_squares)
                if not valid_moves:
                    self.is_checkmate = True
            else:
                valid_moves = self.generate_pawn_moves()
                valid_moves.extend(self.generate_king_moves())
                for p in range(2, 6):
                    valid_moves.extend(self.generate_QRBN_moves(piece_type=p))
                # for square in self.squares_of_player:
                #     valid_moves.extend(self.generate_moves_for_square(square))
                if not valid_moves:
                    self.is_draw = True
            self.valid_moves = valid_moves
        return

    @property
    def is_dead_position(self):
        return

    def generate_moves_for_square(self, s0: np.ndarray) -> list[Move]:
        piece_type = self.piece_types(ps=self.piece_in_squares(ss=s0))
        if piece_type == 0:
            return []

        move_dirs = self.MOVE_VECTORS[piece_type] * self.player
        unpinned_mask = self.mask_absolute_pin(s=s0, ds=move_dirs)
        move_dirs_unpinned = move_dirs[unpinned_mask]
        if piece_type == 2:
            end_squares = s0 + move_dirs_unpinned
            inboards = end_squares[self.squares_are_inside_board(end_squares)]
            vacants = inboards[np.bitwise_not(self.squares_belong_to_player(inboards))]
            return self.generate_move_objects(s0=s0, s1s=vacants)

        neighbor_positions = self.neighbor_squares(s=s0, ds=move_dirs_unpinned)
        neighbor_pieces = self.piece_in_squares(neighbor_positions)
        shift = self.pieces_belong_to_player(ps=neighbor_pieces)
        if piece_type == 1:
            shift[0] = shift[0] | self.pieces_belong_to_opponent(neighbor_pieces[0])
            move_mag_restricted = np.minimum(
                np.abs(neighbor_positions - s0).max(axis=1) - shift,
                self.pawn_move_restriction(s0=s0)[unpinned_mask],
            )
        elif piece_type == 6:
            shift[[2, -2]] = shift[[2, -2]] | self.pieces_belong_to_opponent(
                neighbor_pieces[[2, -2]]
            )

            move_mag_restricted = np.minimum(
                np.abs(neighbor_positions - s0).max(axis=1) - shift,
                self.king_move_restriction[unpinned_mask],
            )
        else:
            move_mag_restricted = np.abs(neighbor_positions - s0).max(axis=1) - shift
        valid_move_mask = move_mag_restricted > 0
        valid_move_mags = move_mag_restricted[valid_move_mask]
        valid_move_dirs = move_dirs_unpinned[valid_move_mask]
        valid_moves = [
            d * np.expand_dims(np.arange(1, mag + 1, dtype=np.int8), 1)
            for mag, d in zip(valid_move_mags, valid_move_dirs)
        ]
        if not valid_moves:
            return []

        valid_moves = np.concatenate(valid_moves, dtype=np.int8)
        valid_s1s = (s0 + valid_moves).reshape((-1, 2))
        if piece_type == 6:
            valid_s1s = valid_s1s[self.king_wont_be_attacked(ss=valid_s1s)]

        return self.generate_move_objects(s0=s0, s1s=valid_s1s, is_pawn=piece_type == 1)

    def generate_QRBN_moves(self, piece_type: int):
        def update_view(mask):
            nonlocal s0s_valid, s1s_valid
            s0s_valid = s0s_valid[mask]
            s1s_valid = s1s_valid[mask]

        def mask_can_move_into():
            if piece_type == 13:
                # For pawn captures
                return self.squares_belong_to_opponent(ss=s1s_valid) | self.is_enpassant_square(
                    ss=s1s_valid
                )
            elif piece_type == 12:
                return s0s_valid[..., 0] == (
                    1 if self.player == 1 else 6
                ) & ~self.squares_belong_to_player(
                    ss=s0s_valid + [1, 0]
                ) & ~self.squares_belong_to_player(
                    ss=s1s_valid
                )
            elif piece_type == 62:
                b_file_empty = np.array([True, True], dtype=np.bool_)
                b_file_empty[0 if self.player == 1 else 1] = self.squares_are_empty(
                    ss=s1s_valid[0 if self.player == 1 else 1] - [0, 1]
                )
                return (
                    self.castling_rights[self.player, [-self.player, self.player]]
                    & self.squares_are_empty(ss=s1s_valid)
                    & self.squares_are_empty(ss=s1s_valid - [[0, -self.player], [0, self.player]])
                    & b_file_empty
                )
            else:
                return ~self.squares_belong_to_player(ss=s1s_valid)

        def check_mask():
            if piece_type == 61:
                return self.king_wont_be_attacked(ss=s1s_valid)
            elif piece_type == 62:
                return self.king_wont_be_attacked(ss=s1s_valid) & self.king_wont_be_attacked(
                    ss=s1s_valid - [[0, -self.player], [0, self.player]]
                )
            else:
                return self.mask_unpinned_absolute_vectorized(s0s=s0s_valid, s1s=s1s_valid)

        s0s = self.squares_of_piece(p=piece_type)
        s1s = s0s[:, np.newaxis] + self.MOVE_VECTORS_PIECE[piece_type] * self.player
        mask_inboard = self.squares_are_inside_board(ss=s1s)
        s0s_valid = np.repeat(s0s, np.count_nonzero(mask_inboard, axis=1), axis=0)
        s1s_valid = s1s[mask_inboard]
        update_view(mask_can_move_into())
        update_view(check_mask())

        if piece_type in [3, 4, 5] and s1s_valid.size != 0:
            ds_valid = s1s_valid - s0s_valid
            # s0s_rep = np.tile(s0s, reps=4).reshape(-1, 2)
            # ds_rep = np.concatenate([self.UNIT_VECTORS_ORTHO for i in range(s0s.size // 2)])
            neighbors_pos = self.neighbor_squares_vectorized(
                ss=s0s_valid,
                ds=ds_valid
                # np.tile(self.UNIT_VECTORS_ORTHO, reps=s0s.size // 2).reshape(-1, 2)
            )
            neighbor_pieces = self.piece_in_squares(neighbors_pos)
            shift_for_own_piece = self.pieces_belong_to_player(ps=neighbor_pieces)
            move_mag_restricted = (
                np.abs(neighbors_pos - s0s_valid).max(axis=-1) - shift_for_own_piece
            )
            mask_valid_moves = move_mag_restricted > 0
            valid_move_mags = move_mag_restricted[mask_valid_moves]
            valid_move_dirs = ds_valid[mask_valid_moves]
            valid_moves = [
                d * np.expand_dims(np.arange(1, mag + 1, dtype=np.int8), 1)
                for mag, d in zip(valid_move_mags, valid_move_dirs)
            ]
            if not valid_moves:
                return []
            np.concatenate(valid_moves, dtype=np.int8)
            update_view(mask_valid_moves)
            s1s_valid = s0s_valid + valid_moves
        is_promotion = np.zeros(shape=s0s_valid.size // 2, dtype=np.bool_)
        return self.generate_move_objects_vectorized(
            s0s=s0s_valid, s1s=s1s_valid, is_promotion=is_promotion
        )

        # s1s = s0s[:, np.newaxis] + self.MOVE_VECTORS_PIECE[4]
        # mask_inboard = self.squares_are_inside_board(ss=s1s)
        # s0s_valid = np.repeat(s0s, np.count_nonzero(mask_inboard, axis=1), axis=0)
        # s1s_valid = s1s[mask_inboard]
        #
        # mask_vacant = ~self.squares_belong_to_player(ss=s1s_valid)
        # s0s_valid = s0s_valid[mask_vacant]
        # s1s_valid = s1s_valid[mask_vacant]
        # unpin_mask = self.mask_unpinned_absolute_vectorized(s0s=s0s_valid, s1s=s1s_valid)
        # s0s_valid = s0s_valid[unpin_mask]
        # s1s_valid = s1s_valid[unpin_mask]
        # is_promotion = np.zeros(shape=s1s_valid.size // 2, dtype=np.bool_)
        # return self.generate_move_objects_vectorized(
        #     s0s=s0s_valid, s1s=s1s_valid, is_promotion=is_promotion
        # )

    def generate_move_objects_vectorized(self, s0s, s1s, is_promotion):
        moves = []
        for s0, s1, p in zip(s0s, s1s, is_promotion):
            moves.extend(
                Move(s0, s1, piece)
                for piece in (np.array([2, 3, 4, 5], dtype=np.int8) if p else [None])
            )
        return moves

    def mask_unpinned_absolute_vectorized(self, s0s, s1s):
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

        kingside_neighbors_squares = self.neighbor_squares_vectorized(
            ss=s0s[~current_unpin_mask],
            ds=s0ks_uv[~current_unpin_mask],
        )
        kingside_neighbors = self.piece_in_squares(ss=kingside_neighbors_squares)
        mask_king_protected = kingside_neighbors != self.king
        current_unpin_mask[~current_unpin_mask] = mask_king_protected

        otherside_neighbors_squares = self.neighbor_squares_vectorized(
            ss=s0s[~current_unpin_mask],
            ds=-s0ks_uv[~current_unpin_mask],
        )
        otherside_neighbors = self.piece_in_squares(ss=otherside_neighbors_squares)
        no_queen = otherside_neighbors != -self.player * 5
        has_orthogonal_dir = s0ks_uv[~current_unpin_mask] == 0
        is_orthogonal = has_orthogonal_dir[..., 0] | has_orthogonal_dir[..., 1]
        no_rooks = otherside_neighbors[is_orthogonal] != -self.player * 4
        no_bishops = otherside_neighbors[~is_orthogonal] != -self.player * 3
        mask_no_pinning = no_queen
        mask_no_pinning[is_orthogonal] &= no_rooks
        mask_no_pinning[~is_orthogonal] &= no_bishops
        current_unpin_mask[~current_unpin_mask] = mask_no_pinning

        return current_unpin_mask

    def neighbor_squares_vectorized(self, ss, ds):
        # dists = np.where(ds == 1, 7 - ss, ss)
        # dists[ds == 0] = 8
        # ds_nearest_edge = dists.min(axis=-1)
        neighbor_squares = np.zeros(shape=ss.shape, dtype=np.int8)
        next_neighbors_pos = ss + ds
        # mask_inboard = self.squares_are_inside_board(ss=next_neighbors_pos)
        not_set = np.ones(ss.size // 2, dtype=np.int8)
        while np.any(not_set):
            mask_inboard = self.squares_are_inside_board(ss=next_neighbors_pos)
            neighbor_squares[~mask_inboard] = (next_neighbors_pos - ds)[~mask_inboard]
            not_set[~mask_inboard] = 0
            inboard_squares = next_neighbors_pos[mask_inboard]
            square_type = self.board[inboard_squares[..., 0], inboard_squares[..., 1]]
            mask_not_empty = square_type != 0
            x = mask_inboard.copy()
            x[x] &= mask_not_empty
            neighbor_squares[x] = inboard_squares[mask_not_empty]
            not_set[x] = 0
            next_neighbors_pos += ds
        return neighbor_squares

    def generate_pawn_moves(self):

        s1s_capture = s0s[:, np.newaxis] + self.MOVE_VECTORS_PIECE[1][2:]

    def pawn_move_restriction(self, s0):
        move_dirs = self.MOVE_VECTORS[1] * self.player
        return np.array(
            [
                1 + self.pawn_not_yet_moved(s0),
                self.pawn_can_capture_square(s0 + move_dirs[1]),
                self.pawn_can_capture_square(s0 + move_dirs[2]),
            ],
            dtype=np.int8,
        )

    def pawn_not_yet_moved(self, s0):
        return s0[0] == (1 if self.player == 1 else 6)

    def is_enpassant_square(self, ss):
        return np.all(ss == [(5 if self.player == 1 else 2), self.enpassant_file], axis=-1)

    def pawn_can_capture_square(self, s1):
        can_capture_enpassant = np.all(s1 == [(5 if self.player == 1 else 2), self.enpassant_file])
        can_capture_normal = self.squares_are_inside_board(
            ss=s1
        ) and self.squares_belong_to_opponent(ss=s1)
        return can_capture_normal or can_capture_enpassant

    @property
    def king_move_restriction(self):
        move_restriction = np.ones(8, dtype=np.int8)
        move_restriction[[2, -2]] += [
            self.castling_right(side=1),
            self.castling_right(side=-1)
            and self.squares_are_empty(ss=np.array([0 if self.player == 1 else 7, 1])),
        ]
        return move_restriction

    def generate_move_objects(self, s0, s1s, is_pawn=False):
        moves = []
        for s1 in s1s:
            moves.extend(
                Move(s0, s1, promoted)
                for promoted in (
                    np.array([2, 3, 4, 5], dtype=np.int8)
                    if is_pawn and s1[0] == (7 if self.player == 1 else 0)
                    else [None]
                )
            )
        return moves

    def mask_absolute_pin(self, s: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """
        Whether a given number of orthogonal/diagonal directions from a given square are free
        from breaking an absolute pin for the current player.

        Parameters
        ----------
        s : numpy.ndarray(shape=(2,), dtype=numpy.int8)
            Coordinates of the square.
        ds : numpy.ndarray(shape=(n, 2), dtype=numpy.int8)
            A 2d-array of orthogonal/diagonal directions.

        Returns
        -------
        numpy.ndarray(shape=(n,), dtype=numpy.bool_)
            A boolean array that can be used to select the unpinned directions.
        """
        # A move from a given square cannot break a pin, if:
        # (notice that the order matters, i.e. checking later criteria without first having checked
        # all the earlier ones may result in a wrong conclusion)
        # 1. If the square-king (sk) vector is not cardinal, then regardless of direction,
        # no move from that square is pinned.
        sk_v, sk_uv, sk_vm, is_cardinal = ArrayJudge.move_dir_mag(s0s=s, s1s=self.pos_king)
        if not is_cardinal:
            return np.ones(shape=ds.shape[0], dtype=np.bool_)
        # 2. If there is a piece between the square and the king, or to the other side
        # of the square, along the sk vector, or the piece on the other side is not attacking
        kingside_neigh, otherside_neigh = self.piece_in_squares(
            ss=self.neighbor_squares(s=s, ds=np.array([sk_uv, -sk_uv]))
        )
        if kingside_neigh != self.king or np.isin(
            otherside_neigh, np.array([5, 4 if 0 in sk_uv else 3]) * -self.player, invert=True
        ):
            return np.ones(shape=ds.shape[0], dtype=np.bool_)
        # 3. Otherwise, only directions along the sk-vector are not pinned.
        return np.all(ds == sk_uv, axis=1) | np.all(ds == -sk_uv, axis=1)

    @property
    def pos_king(self):
        return self.squares_of_piece(self.king)[0]

    @property
    def king(self):
        return self.player * 6

    def squares_checking(
        self, s: Optional[np.ndarray] = None, p: Optional[np.int8] = None
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
        numpy.ndarray(shape=(n, 2), dtype=numpy.int8)]
            Coordinates of the 'checking' squares.
        """
        p = -self.player if p is None else p
        s = self.squares_of_piece(6 * -p)[0] if s is None else s
        # 1. CHECK FOR KNIGHT ATTACKS
        # Add given start-square to all knight vectors to get all possible attacking positions
        knight_pos = s + self.MOVE_VECTORS[2]
        # Take those end squares that are within the board
        inboards = knight_pos[self.squares_are_inside_board(ss=knight_pos)]
        mask_knight = self.board[inboards[:, 0], inboards[:, 1]] == p * 2
        # 2. CHECK FOR STRAIGHT-LINE ATTACKS (queen, bishop, rook, pawn, king)
        # Get nearest neighbor in each direction
        neighbors_pos = self.neighbor_squares(s=s, ds=self.DIRECTION_UNIT_VECTORS)
        neighbors = self.piece_in_squares(neighbors_pos)
        # Set an array of opponent's pieces (intentionally add 0 for easier indexing)
        opp_pieces = p * np.arange(7, dtype=np.int8)
        # For queen, rook and bishop, if they are in neighbors, then it means they are attacking
        mask_king = neighbors == opp_pieces[6]
        mask_queen = neighbors == opp_pieces[5]
        mask_rook = neighbors[::2] == opp_pieces[4]
        mask_bishop = neighbors[1::2] == opp_pieces[3]
        mask_pawn = neighbors[1::2] == opp_pieces[1]
        if np.any(mask_pawn):
            dist_vecs = neighbors_pos[1::2] - s
            pawn_restriction = dist_vecs[:, 0] == -p
            mask_pawn = pawn_restriction & mask_pawn
        if np.any(mask_king):
            dist_vecs = neighbors_pos - s
            king_restriction = np.abs(dist_vecs).max(axis=1) == 1
            mask_king = king_restriction & mask_king
        attacking_positions = np.concatenate(
            [
                inboards[mask_knight],
                neighbors_pos[mask_king],
                neighbors_pos[mask_queen],
                neighbors_pos[::2][mask_rook],
                neighbors_pos[1::2][mask_bishop],
                neighbors_pos[1::2][mask_pawn],
            ]
        )
        return attacking_positions

    def squares_attacking(self, s: np.ndarray, p: Optional[np.int8] = None) -> np.ndarray:
        squares_checking = self.squares_checking(s=s, p=p)
        unpin_mask = []
        for square in squares_checking:
            move_v, move_uv, move_vm, is_cardinal = ArrayJudge.move_dir_mag(s0s=square, s1s=s)
            unpin_mask.append(self.mask_absolute_pin(s=square, ds=np.expand_dims(move_uv))[0])
        return squares_checking[unpin_mask]

    def king_wont_be_attacked(self, ss: np.ndarray):
        king_pos = tuple(self.pos_king)
        self.board[king_pos] = 0  # temporarily remove king from board
        square_is_not_attacked = [self.squares_checking(s=square).size == 0 for square in ss]
        self.board[king_pos] = self.king  # put it back
        return np.array(square_is_not_attacked, dtype=np.bool_)

    def moves_resolving_check(self, attacking_squares: np.ndarray):
        # Get the squares the king can move into to resolve check.
        king_pos = self.pos_king
        possible_squares = king_pos + self.DIRECTION_UNIT_VECTORS
        inboard_squares = possible_squares[self.squares_are_inside_board(ss=possible_squares)]
        vacant_squares = inboard_squares[
            np.bitwise_not(self.squares_belong_to_player(inboard_squares))
        ]
        safe_squares = vacant_squares[self.king_wont_be_attacked(ss=vacant_squares)]
        resolving_moves = [
            Move(start_square=king_pos, end_square=safe_square) for safe_square in safe_squares
        ]
        # In case of single checks, get the moves that block or capture the attacking piece.
        if attacking_squares.shape[0] == 1:
            # Find capturing moves
            attacking_positions = self.squares_attacking(s=attacking_squares[0], p=self.player)
            unpinned_mask = self.mask_absolute_pin(
                s=attacking_squares[0], ds=attacking_squares[0] - attacking_positions
            )
            available_capturing_positions = attacking_positions[unpinned_mask]
            for capturing_position in available_capturing_positions:
                resolving_moves.append(
                    Move(start_square=capturing_position, end_square=attacking_squares[0])
                )
            # Find blocking moves
            squares_in_between = self.squares_in_between(s0=attacking_squares[0], s1=king_pos)
            for square in squares_in_between:
                attacking_positions = self.squares_attacking(s=square, p=self.player)

        return resolving_moves

    def neighbor_squares(self, s: np.ndarray, ds: np.ndarray) -> np.ndarray:
        neighbor_positions = []
        for d in ds:
            neighbor_positions.append(self.neighbor_square(s=s, d=d))
        return np.array(neighbor_positions)

    def neighbor_square(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Coordinates of the nearest neighbor to a given square, in a given cardinal direction.

        Parameters
        ----------
        s : numpy.ndarray(shape=(2,), dtype=np.int8)
            Coordinates of the square.
        d : numpy.ndarray(shape=(2,), dtype=np.int8)
            Direction from that square.
            For example, `[1, -1]` means top-left (diagonal), and `[1, 0]` means top.

        Returns
        -------
        numpy.ndarray
            Coordinates of the nearest neighbor in the given direction.
            If the given square is the last square on the board in the given direction, then
            the coordinates of the square itself is returned. On the other hand, if there is
            no piece in the given direction (but the square is not the last),
            then coordinates of the last square in that direction is returned.
        """
        # Calculate distance to the nearest relevant edge. For orthogonal directions, this is the
        # distance to the edge along that direction. For diagonal directions, this is the
        # minimum of the distance to each of the two edges along that direction.
        d_edge = np.where(d == 1, 7 - s, s)[d != 0].min()
        # Get the square-indices that lie in that direction, up until the edge
        squares_along_d = self.squares_in_between(s0=s, s1=s + (d_edge + 1) * d)
        if squares_along_d.size == 0:
            return s
        # Get the corresponding square occupancies.
        sub_board = self.board[squares_along_d[:, 0], squares_along_d[:, 1]]
        # Get indices of non-zero elements (i.e. non-empty squares) in the given direction
        neighbors_idx = np.nonzero(sub_board)[0]
        # If there are no neighbors in that direction, the index array will be empty. In this case
        # we return the position of the last empty square in that direction. Otherwise,
        # the first element corresponds to the index of nearest neighbor in that direction.
        return squares_along_d[neighbors_idx[0] if neighbors_idx.size != 0 else -1]

    @staticmethod
    def squares_in_between(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
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
        in_between_squares = (
            s0
            + np.arange(
                1,
                move_vm if is_cardinal else 1,  # If not on a cardinal ray, make the range empty
                dtype=np.int8,
            )[:, np.newaxis]
            * move_uv
        )
        return in_between_squares

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

    @property
    def move_is_promotion(self) -> bool:
        return

    @property
    def board_is_checkmate(self) -> bool:
        return

    @property
    def board_is_draw(self) -> bool:
        return

    @property
    def game_over(self) -> bool:
        return self.board_is_checkmate or self.board_is_draw

    def player_is_checked(self) -> bool:
        pass

    def castling_right(self, side: int) -> bool:
        """
        Whether current player has castling right for the given side.

        Parameters
        ----------
        side : int
            +1 for kingside, -1 for queenside.
        """
        return self.castling_rights[self.player, side]

    def piece_in_squares(self, ss: np.ndarray) -> Union[np.ndarray, np.int8]:
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
        return self.board[ss[..., 0], ss[..., 1]]

    def squares_of_piece(self, p: int):
        return np.argwhere(self.board == p)

    @property
    def squares_of_player(self):
        return np.argwhere(np.sign(self.board) == self.player)

    def squares_belong_to_player(self, ss: np.ndarray) -> bool:
        """
        Whether a given square has a piece on it belonging to the player in turn.
        """
        return np.sign(self.piece_in_squares(ss=ss)) == self.player

    def squares_belong_to_opponent(self, ss: np.ndarray) -> bool:
        """
        Whether a given square has a piece on it belonging to the opponent.
        """
        return np.sign(self.piece_in_squares(ss=ss)) == self.player * -1

    def squares_are_empty(self, ss: np.ndarray):
        return self.piece_in_squares(ss=ss) == 0

    def pieces_belong_to_player(
        self, ps: Union[np.int8, np.ndarray]
    ) -> Union[np.bool_, np.ndarray]:
        return np.sign(ps) == self.player

    def pieces_belong_to_opponent(
        self, ps: Union[np.int8, np.ndarray]
    ) -> Union[np.bool_, np.ndarray]:
        return np.sign(ps) == self.player * -1

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
        match piece_type:
            case 1:
                return (move_vect[0] == p and move_abs[1] < 2) or np.all(move_vect == [2 * p, 0])
            case 2:
                return not (move_manhattan_dist != 3 or np.isin(3, move_abs))
            case 3:
                return move_abs[0] == move_abs[1]
            case 4:
                return np.isin(0, move_abs)
            case 5:
                return move_abs[0] == move_abs[1] or np.isin(0, move_abs)
            case 6:
                return move_manhattan_dist == 1 or (move_manhattan_dist == 2 and move_abs[0] != 2)
