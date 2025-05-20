import csv
import os
from game_state import GameState
from buttons import Buttons

class DataCollector:
    def __init__(self):
        self.filename = "training_data_20250508_155116.csv"
        self.headers = [
            # Player 1 state
            'p1_health', 'p1_x', 'p1_y', 'p1_is_jumping', 'p1_is_crouching', 'p1_is_player_in_move', 'p1_move_id',
            # Player 2 state
            'p2_health', 'p2_x', 'p2_y', 'p2_is_jumping', 'p2_is_crouching', 'p2_is_player_in_move', 'p2_move_id',
            # Game state
            'timer', 'fight_result', 'has_round_started', 'is_round_over',
            # Button presses (1 for pressed, 0 for not pressed)
            'up', 'down', 'left', 'right', 'Y', 'B', 'A', 'X', 'L', 'R', 'start', 'select'
        ]
        self._verify_csv()
        self.is_recording = False
        print(f"Data collector initialized. Using existing file {self.filename}")

    def _verify_csv(self):
        """Verify that the CSV file exists and has headers."""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print("CSV file not found, created new file with headers.")
        else:
            print("CSV file found. Appending data to existing file.")

    def record_state(self, game_state: GameState, buttons: Buttons):
        """Record the current game state and button presses to CSV."""
        if game_state.has_round_started and not self.is_recording:
            self.is_recording = True
            print("Round started - beginning data collection")

        if game_state.is_round_over and self.is_recording:
            self.is_recording = False
            print("Round ended - stopping data collection")
            return

        if not self.is_recording:
            return

        print("\nRecording game state:")
        print(f"P1 Health: {game_state.player1.health}, Position: ({game_state.player1.x_coord}, {game_state.player1.y_coord})")
        print(f"P2 Health: {game_state.player2.health}, Position: ({game_state.player2.x_coord}, {game_state.player2.y_coord})")
        print(f"Timer: {game_state.timer}, Round Started: {game_state.has_round_started}, Round Over: {game_state.is_round_over}")

        row = [
            game_state.player1.health,
            game_state.player1.x_coord,
            game_state.player1.y_coord,
            int(game_state.player1.is_jumping),
            int(game_state.player1.is_crouching),
            int(game_state.player1.is_player_in_move),
            game_state.player1.move_id,
            game_state.player2.health,
            game_state.player2.x_coord,
            game_state.player2.y_coord,
            int(game_state.player2.is_jumping),
            int(game_state.player2.is_crouching),
            int(game_state.player2.is_player_in_move),
            game_state.player2.move_id,
            game_state.timer,
            game_state.fight_result,
            int(game_state.has_round_started),
            int(game_state.is_round_over),
            int(buttons.up),
            int(buttons.down),
            int(buttons.left),
            int(buttons.right),
            int(buttons.Y),
            int(buttons.B),
            int(buttons.A),
            int(buttons.X),
            int(buttons.L),
            int(buttons.R),
            int(buttons.start),
            int(buttons.select)
        ]

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            print("State recorded to CSV")
