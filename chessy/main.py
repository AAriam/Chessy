import argparse

from chessy.user_interface.gui.main import GraphicalInterface


parser = argparse.ArgumentParser(description="CHESSY")

parser.add_argument(
    "--interface",
    help="Type of interface for the game.",
    type=str,
    choices=["gui", "text", "web_browser"],
    default="gui",
)
parser.add_argument(
    "--num_players",
    help="Number of human players in the game.",
    type=int,
    choices=[0, 1, 2],
    default=2,
)
# parser.add_argument(
#     "--side",
#     help="Which side to play.",
#     type=str,
#     choices=["white", "black"],
#     default="white",
# )
parser.add_argument(
    "--initial_state",
    help="FEN record of the initial position. Default is the standard start position.",
    type=str,
    default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
)
parser.add_argument(
    "--time_left_white",
    help="Remaining time (in minutes) for white. Set to 0 for turning off player's timer.",
    type=float,
    default=0.,
)
parser.add_argument(
    "--time_left_black",
    help="Remaining time (in minutes) for black. Set to 0 for turning off player's timer.",
    type=float,
    default=0.,
)
parser.add_argument(
    "--undo_allowed",
    help="Whether undoing moves is allowed.",
    type=bool,
    choices=[True, False],
    default=False,
)

INTERFACE_CLASS = {
    "gui": GraphicalInterface
}

if __name__ == "__main__":
    args = parser.parse_args()
    interface_type = vars(args).pop("interface")
    INTERFACE_CLASS[interface_type](**vars(args))
