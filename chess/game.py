from typing import Sequence, Tuple, Optional
import numpy as np


class ChessGame:
    def move(self, s0: Tuple[int, int], s1: Tuple[int, int], promote_to: int = 5) -> Optional[int]:
        self._update_game_state(
            s0=s0, s1=s1, promote_to=promote_to
        )  # Otherwise, apply the move and update state
        return

    def _update_game_state(self, s0: np.ndarray, s1: np.ndarray, promote_to: int):

        move = s1 - s0
        move_abs = np.abs(move)
        move_dir = np.sign(move)
        piece = abs(self.piece_in_square(s0))

        if piece == 1:
            if np.all(move_abs == [2, 0]):
                if s0[0] != (1 if self._turn == 1 else 6):
                    raise IllegalMoveError("Pawn cannot move two squares after its first move.")
                if self.piece_in_square(s1) != 0:
                    raise IllegalMoveError("The end-square is occupied.")
                if (
                    s1[1] < 7
                    and self.piece_in_square(s1 + [0, 1]) == -self._turn
                    or s1[1] > 0
                    and self.piece_in_square(s1 + [0, -1]) == -self._turn
                ):
                    self._enpassant = s1[1]
            elif np.all(move_abs == [1, 0]):
                if self.piece_in_square(s1) != 0:
                    raise IllegalMoveError("The end-square is occupied.")
            else:
                if self.piece_in_square(s1) == 0:
                    if self._enpassant != s1[1]:
                        raise IllegalMoveError("Pawn can only move diagonally when capturing.")
                    else:
                        self._board[s0[0], s1[1]] = 0  # Capture opponent's pawn enpassant
                        self._enpassant = -1  # Reset enpassant allowance
            if s1[0] == (7 if self._turn == 1 else 0):
                self._board[tuple(s0)] = self._turn * promote_to
            self._fifty_move_draw_count = 0
        else:
            self._enpassant = -1  # If in this move no pawn is moved, then enpassant resets.
            if piece == 6:
                if move_abs[1] == 2:  # Castling
                    if not self._can_castle[self._turn, move_dir[1]]:
                        raise IllegalMoveError("Castling is not allowed.")
                    if self.piece_in_square(s1) != 0 or (
                        move_dir[1] == -1 and self.piece_in_square(s1 + move_dir) != 0
                    ):
                        raise IllegalMoveError("Castling is blocked.")
                    if self.square_is_attacked_by_opponent(s0 + move_dir):
                        raise IllegalMoveError("Castling way is under attack.")
                    self._board[s0[0], 0 if move_dir[1] == -1 else 7] = 0  # Move the rook
                    self._board[s0[0], s0[1] + move_dir[1]] = 4 * self._turn

                self._can_castle[self._turn] = 0  # Turn off castling allowance
            elif piece == 4:
                if s0[0] == 0:
                    self._can_castle[self._turn, -1] = 0
                if s0[0] == 7:
                    self._can_castle[self._turn, 1] = 0

        if np.sign(self.piece_in_square(s1)) == -self._turn:  # If capture has been made
            self._fifty_move_draw_count = 0  # reset fifty move count
        self._board[tuple(s1)] = self._board[tuple(s0)]
        self._board[tuple(s0)] = 0
        self._turn *= -1
        return

    def move_doesnt_resolve_check(self, s0: np.ndarray, s1: np.ndarray) -> bool:
        king_pos = np.argwhere(self._board == 6 * self._turn)[0]  # Find position of player's king
        if not self.square_is_attacked_by_opponent(king_pos):
            return False

        return True



    def move_is_blocked(self, s0: np.ndarray, s1: np.ndarray) -> bool:
        """
        Whether a straight-line way from start-square to end-square is blocked by any piece.
        Does not work for knights (and doesn't need to).
        """
        if abs(self.piece_in_square(s=s0)) == 2:
            return False
        if self.square_belongs_to_current_player(s=s1):
            return True
        move_dir = np.sign(s1 - s0)
        neighbor, pos_neighbor = self.neighbor_in_direction(s=s0, d=move_dir)
        dif = pos_neighbor - s1
        return np.all(np.sign(dif) == -move_dir)
        # neighbor_manhattan_dist = np.abs(pos_neighbor - s0).sum()
        # dif = neighbor_manhattan_dist - move_manhattan_dist
        # # Move is blocked when nearest neighbor is closer than the end-square
        # if neighbor_manhattan_dist >= move_manhattan_dist:
        #     return False, ""
        # block_color = self._COLORS[np.sign(neighbor)]
        # block_piece = self._PIECES[abs(neighbor)]
        # return True, f"Move is blocked by {block_color}'s {block_piece} at {pos_neighbor}"

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
            opp_pieces[5] in neighbors
            or opp_pieces[4] in neighbors[:4]  # Queen attacking
            or opp_pieces[3] in neighbors[4:]  # Rook attacking  # Bishop attacking
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
                    (piece == opp_pieces[1] and dist_vec[0] == self._turn)
                    or
                    # Piece is king, and it's one square away in any direction
                    (piece == opp_pieces[6] and np.abs(dist_vec).max() == 1)
                ):
                    return True
        # All criteria are checked, return False
        return False

    def all_neighbors(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pieces = np.empty(shape=8, dtype=np.int8)
        coords = np.empty(shape=(8, 2), dtype=np.int8)
        for idx, direction in enumerate(self._DIRECTION_UNIT_VECTORS):
            pieces[idx], coords[idx] = self.neighbor_in_direction(s, direction)
        return pieces, coords
