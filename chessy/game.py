
import time
from chessy.board_representation import BoardState, Player, Move
from chessy.judges.square_list import ArrayJudge
from chessy.judges.abc import IllegalMoveError, GameOverError

class GameError(Exception):
    pass


class Game:

    def __init__(
            self,
            initial_state: BoardState,
            time_left_white: float = 0,
            time_left_black: float = 0,
    ):
        if not isinstance(initial_state, BoardState):
            raise TypeError("Argument `initial_state` must be a `BoardState` type.")
        if not isinstance(time_left_white, float):
            raise TypeError("Argument `time_left_white` must be a `float` type.")
        if not isinstance(time_left_black, float):
            raise TypeError("Argument `time_left_black` must be a `float` type.")
        if time_left_white < 0:
            raise ValueError("Argument `time_left_white` cannot be negative.")
        if time_left_black < 0:
            raise ValueError("Argument `time_left_black` cannot be negative.")

        self.is_clocked_white: bool = time_left_white != 0
        self.is_clocked_black: bool = time_left_black != 0
        self._timer_white: float = time_left_white
        self._timer_black: float = time_left_black

        self.judge: ArrayJudge = ArrayJudge(initial_state)

        self._game_history: list[BoardState] = [initial_state]

        self._current_state: int = -1
        self._timer_global: float = 0
        return

    @property
    def current_state(self):
        return self._game_history[self._current_state]

    @property
    def current_player(self):
        return self.current_state.player

    def drop_clock(self):
        self._timer_curr_player_curr_move = time.time()
        return

    def submit_move(self, move: Move):
        if self._current_state != -1:
            self._game_history = self._game_history[0:self._current_state + 1]
            self.judge = ArrayJudge(self._game_history[-1])
        try:
            self.judge.submit_move(move=move)
        except (IllegalMoveError, GameOverError):
            raise
        else:
            self._game_history.append(self.judge.current_state)


    @property
    def timer(self):
        return

    @property
    def ply_count(self):
        return len(self._game_history) - 1

    def undo(self):
        if abs(self._current_state) < len(self._game_history):
            self._current_state -= 1
        else:
            raise GameError("There is no earlier state.")
        return

    def redo(self):
        if self._current_state < -1:
            self._current_state += 1
        else:
            raise GameError("There is no next state.")
        return

    def jump_to_ply(self, ply_num: int):
        if 0 <= ply_num <= len(self._game_history):
            self._current_state = ply_num - self.ply_count - 1
        return

