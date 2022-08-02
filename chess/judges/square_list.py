from typing import Optional, Sequence, NamedTuple, Any, NoReturn, Union

import numpy as np

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
        6: np.array(
            [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.int8
        ),
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

        self.state_not_yet_analyzed: bool = True
        self.is_checkmate: bool = False
        self.is_check: bool = False
        self.is_draw: bool = False
        self.valid_moves: list[Move] = []
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
        #if self.state_not_yet_analyzed:
        self.generate_all_valid_moves()
        if self.is_checkmate:
            raise GameOverError("Game over. Current player is checkmated.")
        if self.is_draw:
            raise GameOverError("Game over. It is a draw.")
        self.raise_for_preliminaries(move=move)
        if move not in self.valid_moves:
            if self.is_check:
                raise IllegalMoveError("Move does not resolve check.")
            raise IllegalMoveError("Submitted move is illegal.")
        self.apply_move(move=move)
        return

    def raise_for_preliminaries(self, move: Move) -> NoReturn:
        move_vect = move.end_square - move.start_square
        piece = self.piece_in_squares(ss=move.start_square)
        if not self.squares_are_inside_board(ss=move.start_square):
            raise IllegalMoveError("Start-square is out of board.")
        if not piece:
            raise IllegalMoveError("Start-square is empty.")
        piece_name = PIECE[self.piece_types(piece)].name
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
        return

    def apply_move(self, move: Move) -> None:
        piece_at_end_square = self.piece_in_squares(ss=move.start_square)
        moving_piece_type = self.piece_types(piece_at_end_square)
        captured_piece = self.piece_in_squares(ss=move.end_square)
        if captured_piece != 0:
            self.fifty_move_count = 0
        move_vec = move.end_square - move.start_square
        move_vec_mag = np.abs(move_vec)
        if moving_piece_type == 1:
            self.fifty_move_count = 0
            if move.promote_to is not None:
                piece_at_end_square = move.promote_to * self.player
            if np.all(move_vec_mag == [1, 1]) and captured_piece == 0:
                self.board[move.end_square[0] - self.player, move.end_square[1]] = 0
            if move_vec_mag[0] == 2:
                self.enpassant_file = move.end_square[1]
        else:
            self.enpassant_file = -1
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
        self.player *= -1
        self.state_not_yet_analyzed = True
        return

    def generate_all_valid_moves(self) -> list[Move]:
        cheking_squares = self.check_status
        if cheking_squares.size != 0:
            self.is_check = True
            valid_moves = self.moves_resolving_check(attacking_squares=cheking_squares)
            if not valid_moves:
                self.is_checkmate = True
        else:
            valid_moves = []
            for square in self.squares_of_player:
                valid_moves.extend(self.generate_moves_for_square(square))
            if not valid_moves:
                self.is_draw = True
        self.valid_moves = valid_moves
        self.state_not_yet_analyzed = False
        return valid_moves

    def generate_moves_for_square(self, s0: np.ndarray) -> list[Move]:
        piece = self.piece_in_squares(ss=s0)
        piece_type = self.piece_types(pp=piece)
        if piece_type == 0:
            return []
        move_dirs = self.MOVE_VECTORS[piece_type] * self.player
        unpinned_mask = self.mask_absolute_pin(s0=s0, ds=move_dirs)
        move_dirs_unpinned = move_dirs[unpinned_mask]
        if piece_type == 2:
            end_squares = s0 + move_dirs_unpinned
            inboards = end_squares[self.squares_are_inside_board(end_squares)]
            vacants = inboards[np.bitwise_not(self.squares_belong_to_player(inboards))]
            moves = []
            for vacant in vacants:
                moves.append(Move(start_square=s0, end_square=vacant))
            return moves

        neighbor_positions = self.neighbor_square(s=s0, ds=move_dirs_unpinned)
        move_mag_max = np.abs(neighbor_positions - s0).max(axis=1)

        neighbor_pieces = self.piece_in_squares(neighbor_positions)
        shift = self.pieces_belong_to_player(ps=neighbor_pieces)

        if piece_type == 1:
            shift[0] = np.bitwise_or(shift[0], self.pieces_belong_to_opponent(neighbor_pieces[0]))
        if piece_type == 6:
            shift[[2, -2]] = np.bitwise_or(
                shift[[2, -2]], self.pieces_belong_to_opponent(neighbor_pieces[[2, -2]])
            )

        move_mag_restricted = move_mag_max - shift

        if piece_type == 1:
            move_restriction = np.array(
                [
                    2 if s0[0] == (1 if self.player == 1 else 6) else 1,
                    1
                    if (
                        s0[1] + move_dirs[1, 1] == self.enpassant_file
                        or (
                            self.squares_are_inside_board(ss=s0 + move_dirs[1])
                            and self.squares_belong_to_opponent(ss=s0 + move_dirs[1])
                        )
                    )
                    else 0,
                    1
                    if (
                        s0[1] + move_dirs[2, 1] == self.enpassant_file
                        or (
                            self.squares_are_inside_board(ss=s0 + move_dirs[2])
                            and self.squares_belong_to_opponent(ss=s0 + move_dirs[2])
                        )
                    )
                    else 0,
                ],
                dtype=np.int8,
            )[unpinned_mask]
            move_mag_restricted = np.minimum(move_mag_restricted, move_restriction)
        elif piece_type == 6:
            move_restriction = np.ones(8, dtype=np.int8)
            move_restriction[[2, -2]] += [
                self.castling_right(side=1),
                self.castling_right(side=-1)
                and self.squares_are_empty(ss=np.array([0 if self.player == 1 else 7, 0])),
            ]
            move_restriction = move_restriction[unpinned_mask]
            move_mag_restricted = np.minimum(move_mag_restricted, move_restriction)

        valid_move_mask = move_mag_restricted > 0
        valid_move_mags = move_mag_restricted[valid_move_mask]
        valid_move_dirs = move_dirs_unpinned[valid_move_mask]

        valid_moves = []
        for mag, d in zip(valid_move_mags, valid_move_dirs):
            valid_moves.append((d * np.expand_dims(np.arange(1, mag + 1, dtype=np.int8), 1)))
        if not valid_moves:
            return []

        valid_moves = np.concatenate(valid_moves, dtype=np.int8)
        valid_s1s = (s0 + valid_moves).reshape((-1, 2))

        final_move_objs = []
        if piece_type == 1:
            for valid_s1 in valid_s1s:
                if valid_s1[0] == (7 if self.player == 1 else 0):
                    for promoted_piece in [2, 3, 4, 5]:
                        final_move_objs.append(
                            Move(
                                start_square=s0,
                                end_square=valid_s1,
                                promote_to=np.int8(promoted_piece),
                            )
                        )
                else:
                    final_move_objs.append(
                        Move(
                            start_square=s0,
                            end_square=valid_s1,
                        )
                    )
        else:
            if piece_type == 6:
                valid_s1s = valid_s1s[self.king_wont_be_attacked(ss=valid_s1s)]
            for valid_s1 in valid_s1s:
                final_move_objs.append(Move(start_square=s0, end_square=valid_s1))
        return final_move_objs

    def mask_absolute_pin(self, s0: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """
        Whether a given number of orthogonal/diagonal directions from a given square are free
        from breaking an absolute pin for the current player.

        Parameters
        ----------
        s0 : numpy.ndarray(shape=(2,), dtype=numpy.int8)
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
        # 1. If the piece and king are not on the same row, same column, or same diagonal.
        king_pos = self.squares_of_piece(6 * self.player)[0]  # Find position of player's king
        s0_king_vec = king_pos - s0  # Distance vector from start square to king square
        s0_king_vec_abs = np.abs(s0_king_vec)
        if not np.isin(0, s0_king_vec) and s0_king_vec_abs[0] != s0_king_vec_abs[1]:
            return np.ones(shape=ds.shape[0], dtype=np.bool_)

        dir_king = s0_king_vec // s0_king_vec_abs.max()  # King's direction vector
        neighbors_pos = self.neighbor_square(s=s0, ds=np.array([dir_king, -dir_king]))
        neighbors_pieces = self.piece_in_squares(ss=neighbors_pos)
        if (
            neighbors_pieces[0] != 6 * self.player
            or (not self.pieces_belong_to_opponent(neighbors_pieces[1]))
            or neighbors_pieces[1] not in (np.array([5, 4 if 0 in dir_king else 3]) * -self.player)
        ):
            return np.ones(shape=ds.shape[0], dtype=np.bool_)
        return np.bitwise_or(np.all(ds == dir_king, axis=1), np.all(ds == -dir_king, axis=1))

    @property
    def check_status(self):
        return self.attack_status(s=self.squares_of_piece(6 * self.player))

    def attack_status(self, s: np.ndarray, attacking_player: np.int8 = None):
        attacking_player = -self.player if attacking_player is None else attacking_player
        s = np.expand_dims(s, axis=0) if s.ndim == 1 else s
        # 1. CHECK FOR KNIGHT ATTACKS
        # Add given start-square to all knight vectors to get all possible attacking positions
        knight_pos = s + self.MOVE_VECTORS[2]
        # Take those end squares that are within the board
        inboards = knight_pos[self.squares_are_inside_board(ss=knight_pos)]
        knight_presence = self.board[inboards[:, 0], inboards[:, 1]] == attacking_player * 2
        # 2. CHECK FOR STRAIGHT-LINE ATTACKS (queen, bishop, rook, pawn, king)
        # Get nearest neighbor in each direction
        neighbors_pos = self.neighbor_square(s=s, ds=self.DIRECTION_UNIT_VECTORS)
        neighbors = self.piece_in_squares(neighbors_pos)
        # Set an array of opponent's pieces (intentionally add 0 for easier indexing)
        opp_pieces = attacking_player * np.arange(7, dtype=np.int8)
        # For queen, rook and bishop, if they are in neighbors, then it means they are attacking
        king_presence = neighbors == opp_pieces[6]
        queen_presence = neighbors == opp_pieces[5]
        rook_presence = neighbors[::2] == opp_pieces[4]
        bishop_presence = neighbors[1::2] == opp_pieces[3]
        pawn_presence = neighbors[1::2] == opp_pieces[1]
        if np.any(pawn_presence):
            dist_vecs = neighbors_pos[1::2] - s
            pawn_restriction = dist_vecs[:, 0] == -attacking_player
            pawn_presence = np.bitwise_and(pawn_restriction, pawn_presence)
        if np.any(king_presence):
            dist_vecs = neighbors_pos - s
            king_restriction = np.abs(dist_vecs).max(axis=1) == 1
            king_presence = np.bitwise_and(king_restriction, king_presence)
        attacking_positions = np.concatenate(
            [
                inboards[knight_presence],
                neighbors_pos[king_presence],
                neighbors_pos[queen_presence],
                neighbors_pos[::2][rook_presence],
                neighbors_pos[1::2][bishop_presence],
                neighbors_pos[1::2][pawn_presence],
            ]
        )
        # num_attacking_pieces = attacking_positions.shape[0]
        # if num_attacking_pieces == 0:
        #     return None
        return attacking_positions

    def king_wont_be_attacked(self, ss: np.ndarray):
        king_pos = self.squares_of_piece(6 * self.player)[0]
        self.board[tuple(king_pos)] = 0  # temporarily remove king from board
        square_is_not_attacked = [self.attack_status(s=square).size == 0 for square in ss]
        self.board[tuple(king_pos)] = 6 * self.player
        return np.array(square_is_not_attacked, dtype=np.bool_)

    def moves_resolving_check(self, attacking_squares: np.ndarray):

        # Get the squares the king can move into to resolve check. These are the squares
        # that i) don't lie under any current attack rays
        king_pos = self.squares_of_piece(6 * self.player)
        # attack_vects = (attacking_squares - king_pos)[:, np.newaxis]
        # possible_directions = self.DIRECTION_UNIT_VECTORS[
        #     np.all(np.any(self.DIRECTION_UNIT_VECTORS != attack_vects, axis=2), axis=0)
        # ]
        # possible_squares = king_pos + possible_directions
        possible_squares = king_pos + self.DIRECTION_UNIT_VECTORS
        inboard_squares = possible_squares[self.squares_are_inside_board(ss=possible_squares)]
        vacant_squares = inboard_squares[
            np.bitwise_not(self.squares_belong_to_player(inboard_squares))
        ]
        self.board[tuple(king_pos[0])] = 0  # temporarily remove king from board
        resolving_moves = []
        for vacant_square in vacant_squares:
            attacking_positions = self.attack_status(s=vacant_square)
            if attacking_positions.size == 0:
                resolving_moves.append(Move(start_square=king_pos, end_square=vacant_square))
        self.board[tuple(king_pos[0])] = 6 * self.player  # put the king back
        if attacking_squares.shape[0] == 1:
            # Find moves that block or capture the attacking piece.
            attacking_positions = self.attack_status(
                s=attacking_squares, attacking_player=self.player
            )
            unpinned_mask = self.mask_absolute_pin(s0=attacking_squares[0], ds=attacking_squares[0] - attacking_positions)
            available_capturing_positions = attacking_positions[unpinned_mask]
            for capturing_position in available_capturing_positions:
                resolving_moves.append(Move(start_square=capturing_position, end_square=attacking_squares[0]))





        return resolving_moves

    def neighbor_squares(self, s: np.ndarray, ds: np.ndarray) -> np.ndarray:
        neighbor_positions = []
        for d in ds:
            neighbor_positions.append(self.neighbor_square(s=s, d=d))
        return np.array(neighbor_positions)

    def neighbor_square(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Coordinates of the nearest neighbor to a given square, in a given direction.

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
        move = s1 - s0
        move_abs = np.abs(move)
        move_mag = move_abs.max()
        # If not on a diagonal or orthogonal ray
        if move_abs[0] != move_abs[1] and np.isin(0, move_abs, invert=True):
            move_mag = 1  # This will make the range empty, while no DivideByZero error occurs
        move_dir = move // move_mag
        return s0 + np.arange(1, move_mag, dtype=np.int8)[:, np.newaxis] * move_dir


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
        Get the type of piece(s) on given square(s) (or 0 if empty).

        Returns
        -------
        Union[np.ndarray, np.int8]
        """
        single_square = ss.ndim == 1
        ss = np.expand_dims(ss, axis=0) if single_square else ss
        pieces = self.board[ss[:, 0], ss[:, 1]]
        return pieces[0] if single_square else pieces

    def squares_of_piece(self, p: np.int8):
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
          Either the coordinates of a single square (shape (2,)),
          or n squares (shape(n, 2)).

        Returns
        -------
        numpy.ndarray
          A 1d boolean array with same size as number of input squares.
        """
        if ss.ndim == 1:
            ss = np.expand_dims(ss, axis=0)
        return np.all(np.all([ss < 8, ss > -1], axis=0), axis=1)

    @staticmethod
    def piece_types(pp: Union[np.int8, np.ndarray]) -> Union[np.int8, np.ndarray]:
        """
        Piece-type of a number of given pieces.
        Piece-type is the absolute value of a piece data, as defined in `BoardState`, i.e.
        0 = empty, 1 = pawn, 2 = knight, 3 = bishop, 4 = rook, 5 = queen, 6 = king.

        Parameters
        ----------
        pp : Union[np.int8, np.ndarray]
            Piece data as defined by `BoardState`.
        Returns
        -------
        Union[np.int8, np.ndarray]
            Piece-type, either as a single integer, or an array of integers, depending on input.
        """
        return np.abs(pp)

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
