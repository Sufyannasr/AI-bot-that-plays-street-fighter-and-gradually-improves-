import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from buttons import Buttons

class MLModel:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=44
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.movement_threshold = 0.3  # Keep movement threshold
        self.attack_threshold = 0.1    # Very low attack threshold
        self.last_p2_health = None  # Track opponent's health
        self.move_damage = {}  # Store movement patterns
        self.button_damage = {}  # Store attack patterns
        self.last_attack_time = 0  # Track last attack time
        self.attack_cooldown = 2   # Very short cooldown
        self.last_move = None  # Track last move used
        self.move_history = []  # Track recent moves
        self.move_history_size = 5  # Keep track of more moves
        self.attack_attempts = 0  # Track consecutive attack attempts
        self.special_move_cooldown = 0  # Track special move cooldown
        self.projectile_cooldown = 0  # Track projectile cooldown
        
    def prepare_features(self, game_state, player):
        """Extract relevant features from the game state"""
        if player == "1":
            p1 = game_state.player1
            p2 = game_state.player2
        else:
            p1 = game_state.player2
            p2 = game_state.player1
            
        # Calculate distance between players
        distance = abs(p1.x_coord - p2.x_coord)
        
        # Calculate relative health advantage
        health_advantage = p1.health - p2.health
        
        # Calculate attack states
        opponent_attacking = int(p1.is_player_in_move and p1.move_id > 0)
        we_attacking = int(p2.is_player_in_move and p2.move_id > 0)
        
        # Swap p1 and p2 features to match training data perspective
        features = [
            p2.health,  # Swapped
            p2.x_coord,  # Swapped
            p2.y_coord,  # Swapped
            int(p2.is_jumping),  # Swapped
            int(p2.is_crouching),  # Swapped
            int(p2.is_player_in_move),  # Swapped
            p2.move_id,  # Swapped
            p1.health,  # Swapped
            p1.x_coord,  # Swapped
            p1.y_coord,  # Swapped
            int(p1.is_jumping),  # Swapped
            int(p1.is_crouching),  # Swapped
            int(p1.is_player_in_move),  # Swapped
            p1.move_id,  # Swapped
            game_state.timer,
            int(game_state.has_round_started),
            int(game_state.is_round_over),
            distance,  # Distance between players
            health_advantage,  # Health advantage
            opponent_attacking,  # Is opponent attacking?
            we_attacking  # Are we attacking?
        ]
        print(f"Features: {features}")
        return np.array(features).reshape(1, -1)
    
    def prepare_target(self, buttons):
        """Convert button states to a target vector"""
        return np.array([
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
        ])
    
    def train(self, csv_file):
        """Train the model using collected data"""
        if not os.path.exists(csv_file):
            print(f"Training data file {csv_file} not found!")
            return False
            
        # Load and prepare data
        data = pd.read_csv(csv_file)
        print(f"Loaded {len(data)} training examples")
        
        # Calculate additional features
        data['distance'] = abs(data['p1_x'] - data['p2_x'])
        data['health_advantage'] = data['p1_health'] - data['p2_health']
        data['opponent_attacking'] = (data['p1_is_player_in_move'] & (data['p1_move_id'] > 0)).astype(int)
        data['we_attacking'] = (data['p2_is_player_in_move'] & (data['p2_move_id'] > 0)).astype(int)
        
        # Swap p1 and p2 columns to train on Player 2's perspective
        data_swapped = data.copy()
        for col in ['health', 'x', 'y', 'is_jumping', 'is_crouching', 'is_player_in_move', 'move_id']:
            data_swapped[f'p1_{col}'], data_swapped[f'p2_{col}'] = data_swapped[f'p2_{col}'], data_swapped[f'p1_{col}']
        
        # Define feature columns to match prepare_features method
        feature_columns = [
            'p2_health', 'p2_x', 'p2_y', 'p2_is_jumping', 'p2_is_crouching',
            'p2_is_player_in_move', 'p2_move_id', 'p1_health', 'p1_x', 'p1_y',
            'p1_is_jumping', 'p1_is_crouching', 'p1_is_player_in_move', 'p1_move_id',
            'timer', 'has_round_started', 'is_round_over',
            'distance', 'health_advantage', 'opponent_attacking', 'we_attacking'
        ]
        
        target_columns = [
            'up', 'down', 'left', 'right', 'Y', 'B', 'A', 'X', 'L', 'R',
            'start', 'select'
        ]
        
        X = data_swapped[feature_columns].values
        y = data_swapped[target_columns].values
        
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Save the trained model and scaler
        joblib.dump(self.model, 'mlp_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        
        return True
    
    def load_model(self):
        """Load a previously trained model"""
        if os.path.exists('mlp_model.joblib') and os.path.exists('scaler.joblib'):
            self.model = joblib.load('mlp_model.joblib')
            self.scaler = joblib.load('scaler.joblib')
            self.is_trained = True
            return True
        return False
    
    def set_damage_analysis(self, move_damage, button_damage):
        """Store the damage analysis from training"""
        self.move_damage = move_damage
        self.button_damage = button_damage
        print("Damage analysis loaded into model")
        
    def predict(self, game_state, player):
        """Make a prediction for the next action"""
        if not self.is_trained:
            if not self.load_model():
                return None
                
        features = self.prepare_features(game_state, player)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        print(f"Prediction probabilities: {probabilities}")
        
        # Initialize buttons
        buttons = Buttons()
        
        # Get player states
        p1 = game_state.player1 if player == "1" else game_state.player2
        p2 = game_state.player2 if player == "1" else game_state.player2
        
        # Calculate distance
        distance = abs(p1.x_coord - p2.x_coord)
        
        # Update timers
        self.last_attack_time = max(0, self.last_attack_time - 1)
        self.special_move_cooldown = max(0, self.special_move_cooldown - 1)
        self.projectile_cooldown = max(0, self.projectile_cooldown - 1)
        
        # Convert probabilities to button states using thresholds
        buttons.up = probabilities[0] > self.movement_threshold
        buttons.down = probabilities[1] > self.movement_threshold
        buttons.left = probabilities[2] > self.movement_threshold
        buttons.right = probabilities[3] > self.movement_threshold
        
        # Apply successful movement patterns
        if self.move_damage:
            # Find the most successful movement pattern for current distance
            best_pattern = None
            best_score = -1
            
            for pattern, count in self.move_damage.items():
                if count > 5:  # Only consider frequently successful patterns
                    # Calculate pattern score based on distance
                    score = count
                    if distance > 100 and pattern[1]:  # Right movement when far
                        score *= 1.5
                    elif distance < 50 and pattern[0]:  # Left movement when close
                        score *= 1.5
                    
                    if score > best_score:
                        best_score = score
                        best_pattern = pattern
            
            if best_pattern:
                # Apply the pattern with some probability
                if np.random.random() < 0.3:  # 30% chance to use pattern
                    buttons.left = bool(best_pattern[0])
                    buttons.right = bool(best_pattern[1])
                    buttons.up = bool(best_pattern[2])
                    buttons.down = bool(best_pattern[3])
        
        # Force attacks based on distance
        if self.last_attack_time <= 0:
            # Always attack when in range
            if distance < 120:  # Increased attack range
                # Choose attack based on distance
                if distance < 40:  # Close range
                    buttons.Y = True
                    if np.random.random() < 0.5:  # 50% chance for combo
                        buttons.B = True
                        if np.random.random() < 0.3:  # 30% chance for triple combo
                            buttons.A = True
                elif distance < 70:  # Medium range
                    buttons.B = True
                    if np.random.random() < 0.4:  # 40% chance for combo
                        buttons.A = True
                        if np.random.random() < 0.3:  # 30% chance for triple combo
                            buttons.X = True
                elif distance < 100:  # Long range
                    buttons.A = True
                    if np.random.random() < 0.4:  # 40% chance for combo
                        buttons.X = True
                else:  # Very long range
                    buttons.X = True
                
                self.last_attack_time = 1  # Reduced cooldown for more frequent attacks
        
        # Use projectiles at medium to long range
        if self.projectile_cooldown <= 0 and distance > 50:
            if np.random.random() < 0.4:  # 40% chance to use projectile
                buttons.L = True
                if distance < 70:
                    buttons.B = True  # Medium range projectile
                else:
                    buttons.A = True  # Long range projectile
                self.projectile_cooldown = 10  # Longer cooldown for projectiles
        
        # Use special moves more frequently
        if self.special_move_cooldown <= 0:
            if np.random.random() < 0.4:  # 40% chance to use special move
                # Special move inputs based on distance
                if distance < 50:  # Close range special
                    # Forward ↓ ↘ → + Punch (Y)
                    buttons.right = True  # Forward
                    buttons.down = True   # ↓
                    buttons.right = True  # ↘ (right + down)
                    buttons.right = True  # →
                    buttons.Y = True      # Punch
                    if np.random.random() < 0.3:  # 30% chance for special combo
                        buttons.B = True
                elif distance < 80:  # Medium range special
                    # → ↓ ↘ + Punch (B)
                    buttons.right = True  # →
                    buttons.down = True   # ↓
                    buttons.right = True  # ↘ (right + down)
                    buttons.B = True      # Punch
                    if np.random.random() < 0.3:  # 30% chance for special combo
                        buttons.A = True
                else:  # Long range special
                    # Forward ↓ ↘ → + Punch (A)
                    buttons.right = True  # Forward
                    buttons.down = True   # ↓
                    buttons.right = True  # ↘ (right + down)
                    buttons.right = True  # →
                    buttons.A = True      # Punch
                    if np.random.random() < 0.3:  # 30% chance for special combo
                        buttons.X = True
                self.special_move_cooldown = 10  # Reduced cooldown for special moves
        
        # Add jump attacks more frequently
        if not p1.is_jumping and distance < 80:  # Increased range for jump attacks
            if np.random.random() < 0.3:  # 30% chance for jump attack (increased from 0.2)
                buttons.up = True
                if distance < 40:
                    buttons.Y = True
                    if np.random.random() < 0.4:  # 40% chance for jump combo
                        buttons.B = True
                elif distance < 60:
                    buttons.B = True
                    if np.random.random() < 0.4:  # 40% chance for jump combo
                        buttons.A = True
                else:
                    buttons.A = True
                    if np.random.random() < 0.4:  # 40% chance for jump combo
                        buttons.X = True
        
        # If opponent is attacking, be more aggressive with counters
        if p2.is_player_in_move and p2.move_id > 0:
            if distance < 60:  # Increased counter range
                if self.last_attack_time <= 0:  # If not in cooldown
                    # More aggressive counter attacks
                    if distance < 40:
                        buttons.Y = True
                        if np.random.random() < 0.5:  # 50% chance for counter combo
                            buttons.B = True
                    else:
                        buttons.B = True
                        if np.random.random() < 0.4:  # 40% chance for counter combo
                            buttons.A = True
                    self.last_attack_time = 1  # Reduced cooldown for counters
        
        # If opponent is low on health, be more aggressive
        if p2.health < 50 and self.last_attack_time <= 0:
            # Use multiple attacks
            if distance < 30:
                buttons.Y = True
                buttons.B = True
            elif distance < 50:
                buttons.B = True
                buttons.A = True
            else:
                buttons.A = True
                buttons.X = True
            self.last_attack_time = self.attack_cooldown
            
            # Use special moves more frequently
            if self.special_move_cooldown <= 0:
                buttons.R = True
                if distance < 30:
                    buttons.Y = True
                elif distance < 50:
                    buttons.B = True
                else:
                    buttons.A = True
                self.special_move_cooldown = 10  # Shorter cooldown when opponent is low
        
        # If we're low on health, be more defensive but still attack
        if p1.health < 50:
            buttons.down = True  # Block more
            if distance < 40:  # If too close, try to create distance
                buttons.left = p1.x_coord > p2.x_coord
                buttons.right = p1.x_coord < p2.x_coord
            elif self.last_attack_time <= 0:  # Attack when not too close
                if distance < 30:
                    buttons.Y = True
                elif distance < 50:
                    buttons.B = True
                else:
                    buttons.A = True
                self.last_attack_time = self.attack_cooldown
        
        # Ensure at least one movement button if none are pressed
        if not (buttons.left or buttons.right):
            if p1.x_coord < p2.x_coord:
                buttons.right = True
            else:
                buttons.left = True
        
        return buttons 