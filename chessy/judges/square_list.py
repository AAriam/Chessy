"""
Array-based judge and move-generator.
"""

# Standard library
from typing import Optional, Sequence, NamedTuple, Any, NoReturn, Union, Tuple

# 3rd party
import numpy as np

# Self
from .abc import Judge, IllegalMoveError, GameOverError
from ..board_representation import BoardState, Move, COLOR, PIECE


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
    DIRECTION_UNIT_VECTORS = np.array(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.int8
    )

    UNIT_VECTORS_ORTHO = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int8)
    UNIT_VECTORS_DIAG = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int8)

    MOVE_VECTORS_PIECE = {
        1: np.array([[1, 0], [2, 0], [1, 1], [1, -1]], dtype=np.int8),
        11: np.array([[1, 0]], dtype=np.int8),  # Pawn vertical advance
        12: np.array([[2, 0]], dtype=np.int8),  # Pawm double vertical advance
        13: np.array([[1, 1], [1, -1]], dtype=np.int8),  # Pawn attacks
        2: np.array(  # Knight moves
            [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]], dtype=np.int8
        ),
        3: UNIT_VECTORS_DIAG,
        4: UNIT_VECTORS_ORTHO,
        5: np.concatenate([UNIT_VECTORS_ORTHO, UNIT_VECTORS_DIAG]),
        6: np.array(
            [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [0, -2], [0, 2]],
            dtype=np.int8
        ),
        61: DIRECTION_UNIT_VECTORS,  # King normal moves
        62: np.array([[0, -2], [0, 2]], dtype=np.int8),  # King castling moves
    }
    # Squares that must be empty for each player for castling to be allowed. First three squares
    # correspond to queenside castle, and the next two correspond to kingside castle.
    CASTLING_SQUARES_EMPTY = {
        1: np.array([[0, 1], [0, 2], [0, 3], [0, 5], [0, 6]], dtype=np.int8),
        -1: np.array([[7, 1], [7, 2], [7, 3], [7, 5], [7, 6]], dtype=np.int8)
    }
    CASTLING_SQUARES_CHECK = {
        1: np.array([[0, 2], [0, 3], [0, 5], [0, 6]], dtype=np.int8),
        -1: np.array([[7, 2], [7, 3], [7, 5], [7, 6]], dtype=np.int8)
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
        self._valid_moves: list[Move] = []
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

    @property
    def valid_moves(self) -> list[Move]:
        return self._valid_moves

    def submit_move(self, move: Move) -> NoReturn:
        move_vect = move.s1 - move.s0
        piece = self.piece_in_squares(ss=move.s0)
        if self.is_checkmate:
            raise GameOverError("Game over. Current player is checkmated.")
        if self.is_draw:
            raise GameOverError("Game over. It is a draw.")
        if not self.squares_are_inside_board(ss=move.s0):
            raise IllegalMoveError("Start-square is out of board.")
        if not piece:
            raise IllegalMoveError("Start-square is empty.")
        if not self.squares_belong_to_player(ss=move.s0):
            raise IllegalMoveError(f"It is {COLOR[self.player].name}'s turn.")
        if not self.squares_are_inside_board(ss=move.s1):
            raise IllegalMoveError("End-square is out of board.")
        if np.all(move.s0 == move.s1):
            raise IllegalMoveError("Start and end-square are the same.")
        if not self.move_principally_legal_for_piece(p=piece, move_vect=move_vect):
            piece_name = PIECE[self.piece_types(piece)].name
            raise IllegalMoveError(
                f"{piece_name.capitalize()}s cannot move in direction {move_vect}."
            )
        if move not in self._valid_moves:
            if self.is_check:
                raise IllegalMoveError("Move does not resolve check.")
            raise IllegalMoveError("Submitted move is illegal.")
        self.apply_move(move=move)
        return

    def apply_move(self, move: Move) -> None:
        piece_at_end_square = self.piece_in_squares(ss=move.s0)
        moving_piece_type = self.piece_types(piece_at_end_square)
        captured_piece = self.piece_in_squares(ss=move.s1)
        if captured_piece != 0:
            self.fifty_move_count = -1
        move_vec = move.s1 - move.s0
        move_vec_mag = np.abs(move_vec)
        if moving_piece_type == 1:
            # Handle promotions and en passant
            self.fifty_move_count = -1
            if move.p_promo is not None:
                piece_at_end_square = move.p_promo * self.player
            if np.all(move_vec_mag == [1, 1]) and captured_piece == 0:
                self.board[move.s1[0] - self.player, move.s1[1]] = 0
            self.enpassant_file = move.s1[1] if move_vec_mag[0] == 2 else -1
        else:
            self.enpassant_file = -1
            # Apply castling and/or modify castling rights
            if moving_piece_type == 6:
                self.castling_rights[self.player] = 0
                if move_vec_mag[1] == 2:
                    rook_pos = (move.s1[0], 7 if move_vec[1] == 2 else 0)
                    rook_end_pos = (move.s1[0], 5 if move_vec[1] == 2 else 3)
                    self.board[rook_pos] = 0
                    self.board[rook_end_pos] = 4 * self.player
            elif moving_piece_type == 4:
                if move.s0[1] == 0:
                    self.castling_rights[self.player, 1] = 0
                elif move.s0[1] == 7:
                    self.castling_rights[self.player, -1] = 0
        self.board[tuple(move.s1)] = piece_at_end_square
        self.board[tuple(move.s0)] = 0
        self.fifty_move_count += 1
        self.is_check = False
        self.player *= -1
        self.analyze_state()
        return

    def analyze_state(self):
        if self.fifty_move_count == 100 or self.is_dead_position:
            self.is_draw = True
            self._valid_moves = []
        else:
            cheking_squares = self.squares_checking()
            if cheking_squares.size != 0:
                self.is_check = True
                valid_moves = self.generate_valid_moves_checked(attacking_squares=cheking_squares)
                if not valid_moves:
                    self.is_checkmate = True
            else:
                valid_moves = self.generate_valid_moves_unchecked()
                if not valid_moves:
                    self.is_draw = True
            self._valid_moves = valid_moves
        return

    def generate_valid_moves_unchecked(self) -> list[Move]:
        valid_moves = self.generate_pawn_moves()
        valid_moves.extend(self.generate_king_moves())
        valid_moves.extend(self.generate_knight_moves())
        valid_moves.extend(self.generate_big_piece_moves())
        return valid_moves

    def generate_big_piece_moves(self):
        valid_moves = []
        for p in np.array([3, 4, 5], dtype=np.int8):
            s0s = self.squares_of_piece(p=p * self.player)
            s1s = s0s[:, np.newaxis] + self.MOVE_VECTORS_PIECE[p]
            mask_inboard = self.squares_are_inside_board(ss=s1s)
            s0s_valid = np.repeat(s0s, np.count_nonzero(mask_inboard, axis=1), axis=0)
            s1s_valid = s1s[mask_inboard]
            if s1s_valid.size == 0:
                continue
            mask_vacant = ~self.squares_belong_to_player(ss=s1s_valid)
            if not np.any(mask_vacant):
                continue
            mask_unpinned = self.mask_unpinned_absolute_vectorized(s0s=s0s_valid[mask_vacant], s1s=s1s_valid[mask_vacant])
            if not np.any(mask_unpinned):
                continue
            s0s_valid = s0s_valid[mask_vacant][mask_unpinned]
            s1s_valid = s1s_valid[mask_vacant][mask_unpinned]

            ds_valid = s1s_valid - s0s_valid
            neighbors_pos = self.neighbor_squares_vectorized(
                ss=s0s_valid,
                ds=ds_valid
            )
            neighbor_pieces = self.piece_in_squares(neighbors_pos)
            shift_for_own_piece = self.pieces_belong_to_player(ps=neighbor_pieces)
            move_mag_restricted = (
                    np.abs(neighbors_pos - s0s_valid).max(axis=-1) - shift_for_own_piece
            )
            mask_valid_moves = move_mag_restricted > 0
            valid_move_mags = move_mag_restricted[mask_valid_moves]
            valid_move_dirs = ds_valid[mask_valid_moves]
            valid_moves_ = [
                d * np.expand_dims(np.arange(1, mag + 1, dtype=np.int8), 1)
                for mag, d in zip(valid_move_mags, valid_move_dirs)
            ]
            if not valid_moves_:
                continue
            valid_moves_ = np.concatenate(valid_moves_, dtype=np.int8)
            s0s_valid = np.repeat(s0s_valid[mask_valid_moves], valid_move_mags, axis=0)
            s1s_valid = (s0s_valid + valid_moves_).reshape(-1, 2)
            is_promotion = np.zeros(shape=valid_move_mags.sum(), dtype=np.bool_)
            valid_moves.extend(
                self.generate_move_objects(s0s=s0s_valid, s1s=s1s_valid, is_promotion=is_promotion)
            )
        return valid_moves

    def generate_knight_moves(self):
        s0s = self.squares_of_piece(p=self.player * 2)
        if s0s.size == 0:
            return []
        s1s = s0s[:, np.newaxis] + self.MOVE_VECTORS_PIECE[2]
        mask_inboard = self.squares_are_inside_board(ss=s1s)
        s0s_valid = np.repeat(s0s, np.count_nonzero(mask_inboard, axis=1), axis=0)
        s1s_valid = s1s[mask_inboard]
        mask_vacant = ~self.squares_belong_to_player(ss=s1s_valid)
        if not np.any(mask_vacant):
            return []
        mask_unpinned = self.mask_unpinned_absolute_vectorized(s0s=s0s_valid[mask_vacant], s1s=s1s_valid[mask_vacant])
        is_promotion = np.zeros(shape=np.count_nonzero(mask_unpinned), dtype=np.bool_)
        return self.generate_move_objects(
            s0s=s0s_valid[mask_vacant][mask_unpinned],
            s1s=s1s_valid[mask_vacant][mask_unpinned], is_promotion=is_promotion)

    def generate_pawn_moves(self):
        s0s = self.squares_of_piece(p=self.player)
        if s0s.size == 0:
            return []
        s1s_all = s0s[:, np.newaxis] + self.MOVE_VECTORS_PIECE[1] * self.player
        s1s_forward, s1s_attack = s1s_all[:, :2], s1s_all[:, 2:]
        s1s_single, s1s_double = s1s_forward[:, 0], s1s_forward[:, 1]
        running_mask = self.squares_are_inside_board(ss=s1s_all)
        mask_vacant_forward1 = self.squares_are_empty(ss=s1s_single[running_mask[:, 0]])
        mask_vacant_forward2 = self.squares_are_empty(ss=s1s_double[running_mask[:, 1]])
        running_mask[:, 0][running_mask[:, 0]] &= mask_vacant_forward1
        running_mask[:, 1][running_mask[:, 1]] &= mask_vacant_forward2
        if np.any(running_mask[:, 0]):
            unpin_mask_forward = self.mask_unpinned_absolute_vectorized(s0s=s0s[running_mask[:, 0]], s1s=s1s_single[running_mask[:, 0]])
            running_mask[:, 0][running_mask[:, 0]] &= unpin_mask_forward
            running_mask[:, 1] &= running_mask[:, 0]
            mask_in_initial_pos = s0s[..., 0] == (1 if self.player == 1 else 6)
            running_mask[:, 1] &= mask_in_initial_pos
        mask_can_attack = self.squares_belong_to_opponent(
            ss=s1s_attack[running_mask[:, 2:]]) | self.is_enpassant_square(ss=s1s_attack[running_mask[:, 2:]])
        running_mask[:, 2:][running_mask[:, 2:]] &= mask_can_attack
        if np.any(running_mask[:, 2]):
            unpin_mask_attack1 = self.mask_unpinned_absolute_vectorized(
                s0s=s0s[running_mask[:, 2]],
                s1s=s1s_attack[:, 0][running_mask[:, 2]]
            )
            running_mask[:, 2][running_mask[:, 2]] &= unpin_mask_attack1
        if np.any(running_mask[:, 3]):
            unpin_mask_attack2 = self.mask_unpinned_absolute_vectorized(
                s0s=s0s[running_mask[:, 3]],
                s1s=s1s_attack[:, 1][running_mask[:, 3]]
            )
            running_mask[:, 3][running_mask[:, 3]] &= unpin_mask_attack2
        if not np.any(running_mask):
            return []
        is_promotion = s1s_all[running_mask][..., 0] == (7 if self.player == 1 else 0)
        return self.generate_move_objects(s0s=np.repeat(s0s, np.count_nonzero(running_mask, axis=1), axis=0), s1s=s1s_all[running_mask], is_promotion=is_promotion)

    def generate_king_moves(self):
        s1s_all = self.pos_king + self.MOVE_VECTORS_PIECE[6]
        s1s_normal, castle_s1s = s1s_all[:-2], s1s_all[-2:]
        s1s_inboard = s1s_normal[ArrayJudge.squares_are_inside_board(ss=s1s_normal)]
        s1s_vacant = s1s_inboard[~self.squares_belong_to_player(ss=s1s_inboard)]
        s1s_final = s1s_vacant[self.king_wont_be_attacked(ss=s1s_vacant)]
        if not self.is_check and np.any(self.castling_rights[self.player]):
            vacant = self.squares_are_empty(ss=self.CASTLING_SQUARES_EMPTY[self.player])
            mask_vacant = [np.all(vacant[:3]), np.all(vacant[3:])]
            not_checked = self.king_wont_be_attacked(ss=self.CASTLING_SQUARES_CHECK[self.player])
            mask_not_checked = np.all(not_checked.reshape(2, 2), axis=1)
            mask_castle = mask_vacant & mask_not_checked & self.castling_rights[self.player][1:]
            s1s_final_castle = castle_s1s[mask_castle]
            s1s_final = np.concatenate((s1s_final.reshape(-1, 2), s1s_final_castle.reshape(-1, 2)))
        return [Move(s0=self.pos_king, s1=s1) for s1 in s1s_final]

    def generate_valid_moves_checked(self, attacking_squares: np.ndarray):
        # Get the squares the king can move into to resolve check.
        moves = self.generate_king_moves()
        # In case of single checks, get the moves that block or capture the attacking piece.
        if attacking_squares.shape[0] == 1:
            # Find capturing moves
            attacking_positions = self.squares_attacking(s=attacking_squares[0], p=self.player)
            unpinned_mask = self.mask_absolute_pin(
                s=attacking_squares[0], ds=attacking_squares[0] - attacking_positions
            )
            available_capturing_positions = attacking_positions[unpinned_mask]
            moves.extend([Move(s0=capturing_position, s1=attacking_squares[0]) for capturing_position in available_capturing_positions])
            # Find blocking moves
            squares_in_between = self.squares_in_between(s0=attacking_squares[0], s1=self.pos_king)
            for square in squares_in_between:
                attacking_positions = self.squares_attacking(s=square, p=self.player)
        return moves

    def king_wont_be_attacked(self, ss: np.ndarray):
        king_pos = tuple(self.pos_king)
        self.board[king_pos] = 0  # temporarily remove king from board
        square_is_not_attacked = [self.squares_checking(s=square).size == 0 for square in ss]
        self.board[king_pos] = self.king  # put it back
        return np.array(square_is_not_attacked, dtype=np.bool_)

    def squares_attacking(self, s: np.ndarray, p: Optional[np.int8] = None) -> np.ndarray:
        squares_checking = self.squares_checking(s=s, p=p)
        unpin_mask = []
        for square in squares_checking:
            move_v, move_uv, move_vm, is_cardinal = ArrayJudge.move_dir_mag(s0s=square, s1s=s)
            unpin_mask.append(
                self.mask_absolute_pin(s=square, ds=np.expand_dims(move_uv, axis=0))[0]
            )
        return squares_checking[unpin_mask]

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
        knight_pos = s + self.MOVE_VECTORS_PIECE[2]
        # Take those end squares that are within the board
        inboards = knight_pos[self.squares_are_inside_board(ss=knight_pos)]
        mask_knight = self.piece_in_squares(inboards) == p * 2
        # 2. CHECK FOR STRAIGHT-LINE ATTACKS (queen, bishop, rook, pawn, king)
        # Get nearest neighbor in each direction
        neighbors_pos = self.neighbor_squares_vectorized(ss=np.tile(s, 8).reshape(-1, 2), ds=self.DIRECTION_UNIT_VECTORS)
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
            mask_pawn &= (dist_vecs[:, 0] == -p)
        if np.any(mask_king):
            dist_vecs = neighbors_pos - s
            mask_king &= (np.abs(dist_vecs).max(axis=1) == 1)
        checking_squares = np.concatenate(
            [
                inboards[mask_knight],
                neighbors_pos[mask_king],
                neighbors_pos[mask_queen],
                neighbors_pos[::2][mask_rook],
                neighbors_pos[1::2][mask_bishop],
                neighbors_pos[1::2][mask_pawn],
            ]
        )
        return checking_squares

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

    def neighbor_squares_vectorized(self, ss, ds):
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
            mask_occupied = self.piece_in_squares(ss=inboard_squares) != 0
            curr_mask = not_set.copy()
            curr_mask[curr_mask] = mask_occupied
            neighbor_squares[curr_mask] = inboard_squares[mask_occupied]
            not_set[not_set] = ~mask_occupied
            next_neighbors_pos += ds
            # curr_mask = not_set.copy()
            # #current_mask[current_mask] &= ~mask_inboard
            # not_set[not_set] &= mask_inboard
            # curr_mask[curr_mask] &= ~mask_inboard
            # neighbor_squares[curr_mask] = (next_neighbors_pos - ds)[curr_mask]
            # #curr_mask[not_set] &= mask_inboard
            # #not_set[~mask_inboard] = 0
            # inboard_squares = next_neighbors_pos[not_set]
            # square_type = self.board[inboard_squares[..., 0], inboard_squares[..., 1]]
            # mask_not_empty = square_type != 0
            # not_set[not_set] &= ~mask_not_empty
            # #curr_mask[curr_mask] &= mask_not_empty
            # #x = mask_inboard.copy()
            # #x[x] &= mask_not_empty
            # neighbor_squares[curr_mask] = inboard_squares[curr_mask]
            # #not_set[x] = 0
        return neighbor_squares

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

    def pawn_move_restriction(self, s0):
        move_dirs = np.array([[1, 0], [1, 1], [1, -1]], dtype=np.int8) * self.player
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

    @property
    def is_dead_position(self):
        return

    @property
    def pos_king(self):
        return self.squares_of_piece(self.king)[0]

    @property
    def king(self):
        return self.player * 6

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

    @staticmethod
    def generate_move_objects(s0s, s1s, is_promotion):
        moves = []
        for s0, s1, p in zip(s0s, s1s, is_promotion):
            moves.extend(
                [
                    Move(s0, s1, piece)
                    for piece in (np.array([2, 3, 4, 5], dtype=np.int8) if p else [None])
                ]
            )
        return moves