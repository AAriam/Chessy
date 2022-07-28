from typing import Sequence, Tuple, Optional, Union
import numpy as np


class ChessGame:

    _COLORS = {-1: "black", 1: "white"}
    _PIECES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
    _DIRECTION_UNIT_VECTORS = np.array(
        [
            [1, 0],  # top
            [-1, 0],  # bottom
            [0, 1],  # right
            [0, -1],  # left
            [1, 1],  # top-right
            [1, -1],  # top-left
            [-1, 1],  # bottom-right
            [-1, -1],  # bottom-left
        ],
        dtype=np.int8
    )
    # All possible relative moves (i.e. s1 - s0) for a knight
    # Equal to:
    # [vec for vec in itertools.permutations([-2, -1, 1, 2], 2) if abs(vec[0]) + abs(vec[1]) == 3]
    _KNIGHT_VECTORS = np.array(
        [[2, 1], [2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1]],
        dtype=np.int8
    )

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

    def move(self, s0: Tuple[int, int], s1: Tuple[int, int], promote_to: int = 0) -> Optional[int]:
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
        s0 = np.array(s0, dtype=np.int8)
        s1 = np.array(s1, dtype=np.int8)
        if self._game_over:  # Return white's score if the game is over
            return self._score
        self.raise_for_illegal_move(s0=s0, s1=s1)  # Otherwise, raise an error if move is illegal
        self._update_game_state(s0=s0, s1=s1)  # Otherwise, apply the move and update state
        if self._game_over:  # Return white's score if the game is over after the move
            return self._score
        return

    def raise_for_illegal_move(self, s0: np.ndarray, s1: np.ndarray) -> None:
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
        # For move beginning outside the board
        if not self.squares_are_inside_board(ss=s0):
            raise IllegalMoveError("Start-square is out of board.")
        # For moves starting from an empty square
        if not self.piece_in_square(s=s0):
            raise IllegalMoveError("Start-square is empty.")
        # For wrong turn (i.e. it is one player's turn, but other player's piece is being moved)
        if not self.square_belongs_to_current_player(s=s0):
            raise IllegalMoveError(f"It is {self._COLORS[self._turn]}'s turn.")
        # For move ending outside the board
        if not self.squares_are_inside_board(ss=s1):
            raise IllegalMoveError("End-square is out of board.")
        # For move starting and ending in the same square
        if np.all(s0 == s1):
            raise IllegalMoveError("Start and end-square are the same.")
        # For move ending in a square occupied by current player's own pieces
        if self.square_belongs_to_current_player(s=s1):
            raise IllegalMoveError("End-square occupied by current player's piece.")
        is_illegal, msg = self.move_illegal_for_piece(s0=s0, s1=s1)
        if is_illegal:
            raise IllegalMoveError(msg)
        # For move resulting in a self-check
        if self.move_results_in_own_check(s0=s0, s1=s1):
            raise IllegalMoveError("Move results in current player being checked.")
        return

    def move_illegal_for_piece(self, s0: np.ndarray, s1: np.ndarray) -> Tuple[bool, str]:
        move = s1 - s0
        move_abs = move.abs()
        move_dir = move / move_abs.max()
        move_manhattan_dist = move_abs.sum()
        piece = abs(self.piece_in_square(s0))
        match piece:
            case 1:
                cond = (move[0] == 1 and move_abs[1] < 2) or np.all(move == [2, 0])
            case 2:
                cond = not(np.isin(3, move_abs) or move_manhattan_dist != 3)
            case 3:
                cond = move_abs[0] == move_abs[1]
            case 4:
                cond = np.isin(0, move_abs)
            case 5:
                cond = move_abs[0] == move_abs[1] or np.isin(0, move_abs)
            case 6:
                cond = move_manhattan_dist == 1 or (move_manhattan_dist == 2 and move_abs[0] != 2)
        if not cond:
            return True, f"{self._PIECES[piece].capitalize()}s cannot move in direction {move}."
        if piece == 2:
            return False, ""
        neighbor, pos_neighbor = self.neighbor_in_direction(s=s0, d=move_dir)
        neighbor_manhattan_dist = np.abs(pos_neighbor - s0).sum()
        dif = neighbor_manhattan_dist - move_manhattan_dist
        # Move is blocked when nearest neighbor is closer than the end-square
        if neighbor_manhattan_dist >= move_manhattan_dist:
            return False, ""
        block_color = self._COLORS[np.sign(neighbor)]
        block_piece = self._PIECES[abs(neighbor)]
        return True, f"Move is blocked by {block_color}'s {block_piece} at {pos_neighbor}"

    def piece_in_square(self, s: np.ndarray) -> int:
        """
        Type of piece on a given square (or 0 if empty).
        """
        return self._board[tuple(s)]

    def square_belongs_to_current_player(self, s: np.ndarray) -> bool:
        """
        Whether a given square has a piece on it belonging to the player in turn.
        """
        return self._turn == np.sign(self.piece_in_square(s))

    @staticmethod
    def squares_are_inside_board(ss: np.ndarray) -> np.ndarray:
        """
        Whether a number of given squares lie outside the chessboard.

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

    def move_results_in_own_check(self, s0: np.ndarray, s1: np.ndarray) -> bool:
        """
        Whether a given move results in the player making the move to be checked.
        """
        if abs(self.piece_in_square(s0)) == 6:  # If the piece being moved is a king
            return self.square_is_attacked_by_opponent(s=s1)
        return self.move_breaks_absolute_pin(s0=s0, s1=s1)

    def square_is_attacked_by_opponent(self, s: np.ndarray) -> bool:
        """
        Whether a given square is being attacked by one of opponent's pieces.
        """
        # 1. CHECK FOR KNIGHT ATTACKS
        # Add given start-square to all knight vectors to get all possible attacking positions
        knight_pos = s + self._KNIGHT_VECTORS
        # Take those end squares that are within the board
        inboards = knight_pos[self.squares_are_inside_board(ss=knight_pos)]
        # Return True if an opponent's knight (knight = 2) is in one of the squares
        if np.isin(-self._turn * 2, self._board[inboards[:, 0], inboards[:, 1]]):
            return True
        # 2. CHECK FOR STRAIGHT-LINE ATTACKS (queen, bishop, rook, pawn, king)
        # Get nearest neighbor in each direction
        neighbors, idx_neighbors = self.all_neighbors(s=s)
        # Set an array of opponent's pieces (intentionally add 0 for easier indexing)
        opp_pieces = -self._turn * np.arange(7, dtype=np.int8)
        # For queen, rook and bishop, if they are in neighbors, then it means they are attacking
        if (
                opp_pieces[5] in neighbors or  # Queen attacking
                opp_pieces[4] in neighbors[:4] or  # Rook attacking
                opp_pieces[3] in neighbors[4:]  # Bishop attacking
        ):
            return True
        # For king and pawns, we also have to check whether they are in an adjacent square,
        # and for pawns, we also have to check whether they are in an attacking direction
        for piece in (opp_pieces[1], opp_pieces[6]):  # iterate over pawn and king
            # Iterate over indices of the piece, if there are any in neighbors
            for piece_coordinates in idx_neighbors[neighbors == piece]:
                # Calculate distance vector from the square to that piece
                dist_vec = piece_coordinates - s
                if (
                        # Piece is pawn, and it's one square away in an attacking direction
                        (piece == opp_pieces[1] and dist_vec[0] == self._turn) or
                        # Piece is king, and it's one square away in any direction
                        (piece == opp_pieces[6] and np.abs(dist_vec).max() == 1)
                ):
                    return True
        # All criteria are checked, return False
        return False

    def move_breaks_absolute_pin(self, s0: np.ndarray, s1: np.ndarray) -> bool:
        """
        Whether a given move (start-square s0, end-square s1) breaks an absolute pin
        for the current player.
        """
        # A move from a given square cannot break a pin, if:
        # (notice that the order matters, i.e. checking later criteria without first having checked
        # all the earlier ones may result in a wrong conclusion)
        # 1. If the piece and king are not on the same row, same column, or same diagonal.
        king_pos = np.argwhere(self._board == 6 * self._turn)  # Find position of player's king
        s0_king_vec = king_pos - s0  # Distance vector from start square to king square
        s0_king_vec_abs = np.abs(s0_king_vec)
        if not np.isin(0, s0_king_vec) and s0_king_vec_abs[0] != s0_king_vec_abs[1]:
            return False
        # 2. If the move is along the king direction (towards or away from).
        dir_king = s0_king_vec / s0_king_vec_abs.max()  # King's direction vector
        s0_s1_vec = s1 - s0  # Distance vector from start to end square
        dir_move = s0_s1_vec / np.abs(s0_s1_vec).max()  # Move's direction vector
        if np.all(dir_move == dir_king) or np.all(dir_move == -dir_king):
            return False
        # 3. If there is another piece between king and the square.
        kingside_neighbor = self.neighbor_in_direction(s=s0, d=dir_king)[0]
        if kingside_neighbor != 6 * self._turn:
            return False
        # 4. If the immediate neighbor on the other side is not an opponent's piece.
        otherside_neighbor = self.neighbor_in_direction(s=s0, d=-dir_king)[0]
        if otherside_neighbor == 0 or np.sign(otherside_neighbor) == self._turn:
            return False
        # 5. If the opponent's piece cannot attack the king from that direction.
        attacking_pieces = np.array([5, 4 if 0 in dir_king else 3]) * -self._turn
        if otherside_neighbor not in attacking_pieces:
            return False
        # If none of the above criteria is met, then the move does break an absolute pin.
        return True

    def all_neighbors(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        neighbor_pieces = np.empty(shape=8, dtype=np.int8)
        neighbor_coordinates = np.empty(shape=(8, 2), dtype=np.int8)
        for idx, direction in self._DIRECTION_UNIT_VECTORS:
            neighbor_pieces[idx], neighbor_coordinates[idx] = self.neighbor_in_direction(s, direction)
        return neighbor_pieces, neighbor_coordinates

    def neighbor_in_direction(
            self, s: np.ndarray, d: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Get type and coordinates of the nearest neighbor to a given square, in a given direction.

        Parameters
        ----------
        s : numpy.ndarray
            Coordinates of the square.
        d : numpy.ndarray
            Direction from white's perspective, as a unit vector.
            For example, `[1, -1]` means top-left (diagonal), and `[1, 0]` means top.

        Returns
        -------
        Tuple[int, numpy.ndarray]
            Type and coordinates of the nearest neighbor in the given direction.
            If the given square is the last square on the board in the given direction, then
            the type and position of the square itself is returned. On the other hand, if there is
            no piece in the given direction (but the square is not the last),
            then type will be 0 (empty) and position will be the last square in that direction.
        """
        # Calculate distance to the nearest relevant edge. For orthogonal directions, this is the
        # distance to the edge along that direction. For diagonal directions, this is the
        # minimum of the distance to each of the two edges along that direction.
        d_edge = np.where(d == 1, 7 - s, s)[d != 0].min()
        # Slice based on direction and distance to edge, to get the relevant part of the board
        slicing = tuple([slice(s[i] + d[i], s[i] + d[i] * (d_edge + 1), d[i]) for i in range(2)])
        sub_board = self._board[slicing]
        # For diagonal directions, slices are still 2d-arrays, but the slice was done in such a way
        # that all squares diagonal to the given square are now on the main diagonal of the new
        # slice, and can be extracted.
        line = sub_board if 0 in d else np.diagonal(sub_board)
        # Get indices of non-zero elements (i.e. non-empty squares) in the given direction
        neighbors_idx = np.nonzero(line)[0]
        # If there are now neighbors in that direction, the index array will be empty. In this case
        # we return the type and position of the last empty square in that direction. Otherwise,
        # the first element corresponds to the index of nearest neighbor in that direction. To that,
        # add 1 to get distance of square to the nearest piece
        # Based on direction and distance, calculate and return index of the neighbor
        pos = s + d * (d_edge if neighbors_idx.size == 0 else neighbors_idx[0] + 1)
        return self._board[pos[0], pos[1]], pos

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
