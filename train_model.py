from ml_model import MLModel
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def analyze_movement_patterns(data):
    """Analyze successful movement patterns"""
    # Calculate movement success (when movement led to damage)
    data['p1_movement_success'] = (data['p1_damage_dealt'] > 0) & (
        (data['left'] == 1) | (data['right'] == 1) | (data['up'] == 1)
    )
    data['p2_movement_success'] = (data['p2_damage_dealt'] > 0) & (
        (data['left'] == 1) | (data['right'] == 1) | (data['up'] == 1)
    )
    
    # Calculate successful movement combinations
    movement_patterns = {}
    for player in ['p1', 'p2']:
        success_data = data[data[f'{player}_movement_success']]
        for _, row in success_data.iterrows():
            pattern = (
                int(row['left']),
                int(row['right']),
                int(row['up']),
                int(row['down'])
            )
            if pattern not in movement_patterns:
                movement_patterns[pattern] = 0
            movement_patterns[pattern] += 1
    
    return movement_patterns

def analyze_attack_patterns(data):
    """Analyze successful attack patterns"""
    # Calculate attack success (when attack led to damage)
    data['p1_attack_success'] = (data['p1_damage_dealt'] > 0) & (
        (data['Y'] == 1) | (data['B'] == 1) | (data['A'] == 1) | (data['X'] == 1) |
        (data['L'] == 1) | (data['R'] == 1)  # Include special buttons
    )
    data['p2_attack_success'] = (data['p2_damage_dealt'] > 0) & (
        (data['Y'] == 1) | (data['B'] == 1) | (data['A'] == 1) | (data['X'] == 1) |
        (data['L'] == 1) | (data['R'] == 1)  # Include special buttons
    )
    
    # Calculate successful attack combinations
    attack_patterns = {}
    for player in ['p1', 'p2']:
        success_data = data[data[f'{player}_attack_success']]
        for _, row in success_data.iterrows():
            # Include special buttons in pattern
            pattern = (
                int(row['Y']),
                int(row['B']),
                int(row['A']),
                int(row['X']),
                int(row['L']),
                int(row['R'])
            )
            if pattern not in attack_patterns:
                attack_patterns[pattern] = 0
            attack_patterns[pattern] += 1
    
    return attack_patterns

def augment_training_data(data, movement_patterns, attack_patterns):
    """Augment training data with successful patterns"""
    augmented_data = data.copy()
    
    # Add successful movement patterns
    for pattern, count in movement_patterns.items():
        if count > 5:  # Only use frequently successful patterns
            new_rows = data[data['p1_movement_success'] | data['p2_movement_success']].copy()
            new_rows['left'] = pattern[0]
            new_rows['right'] = pattern[1]
            new_rows['up'] = pattern[2]
            new_rows['down'] = pattern[3]
            augmented_data = pd.concat([augmented_data, new_rows])
    
    # Add successful attack patterns
    for pattern, count in attack_patterns.items():
        if count > 5:  # Only use frequently successful patterns
            new_rows = data[data['p1_attack_success'] | data['p2_attack_success']].copy()
            new_rows['Y'] = pattern[0]
            new_rows['B'] = pattern[1]
            new_rows['A'] = pattern[2]
            new_rows['X'] = pattern[3]
            new_rows['L'] = pattern[4]  # Include special buttons
            new_rows['R'] = pattern[5]  # Include special buttons
            augmented_data = pd.concat([augmented_data, new_rows])
    
    # Add special move combinations
    special_moves = [
        # Projectile moves (L button combinations)
        {'L': 1, 'B': 1},  # Common projectile input
        {'L': 1, 'A': 1},  # Another projectile input
        # Special moves (R button combinations)
        {'R': 1, 'Y': 1},  # Special attack
        {'R': 1, 'B': 1},  # Another special
        {'R': 1, 'A': 1},  # Another special
    ]
    
    for move in special_moves:
        new_rows = data.copy()
        for button, value in move.items():
            new_rows[button] = value
        augmented_data = pd.concat([augmented_data, new_rows])
    
    return augmented_data

def train_model():
    # Initialize the ML model
    model = MLModel()
    
    # Find the most recent training data file
    data_files = [f for f in os.listdir('.') if f.startswith('training_data_') and f.endswith('.csv')]
    if not data_files:
        print("No training data files found!")
        return
        
    # Sort by modification time (most recent first)
    data_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = data_files[0]
    
    print(f"Training model using data from {latest_file}")
    
    # Load and analyze the data
    data = pd.read_csv(latest_file)
    print(f"Total training examples: {len(data)}")
    
    # Calculate damage dealt by both players
    data['p1_damage_dealt'] = data['p2_health'].diff().fillna(0).clip(lower=0)
    data['p2_damage_dealt'] = data['p1_health'].diff().fillna(0).clip(lower=0)
    
    # Analyze successful patterns
    print("\nAnalyzing successful movement patterns...")
    movement_patterns = analyze_movement_patterns(data)
    print("\nAnalyzing successful attack patterns...")
    attack_patterns = analyze_attack_patterns(data)
    
    # Augment training data with successful patterns
    print("\nAugmenting training data...")
    augmented_data = augment_training_data(data, movement_patterns, attack_patterns)
    print(f"Total training examples after augmentation: {len(augmented_data)}")
    
    # Calculate additional features
    augmented_data['distance'] = abs(augmented_data['p1_x'] - augmented_data['p2_x'])
    augmented_data['health_advantage'] = augmented_data['p1_health'] - augmented_data['p2_health']
    augmented_data['opponent_attacking'] = (augmented_data['p1_is_player_in_move'] & (augmented_data['p1_move_id'] > 0)).astype(int)
    augmented_data['we_attacking'] = (augmented_data['p2_is_player_in_move'] & (augmented_data['p2_move_id'] > 0)).astype(int)
    
    # Swap p1 and p2 columns to match training perspective
    data_swapped = augmented_data.copy()
    for col in ['health', 'x', 'y', 'is_jumping', 'is_crouching', 'is_player_in_move', 'move_id']:
        data_swapped[f'p1_{col}'], data_swapped[f'p2_{col}'] = data_swapped[f'p2_{col}'], data_swapped[f'p1_{col}']
    
    # Define feature columns
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
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    if model.train(latest_file):
        print("\nModel trained successfully!")
        
        # Evaluate model performance
        y_pred = model.model.predict(X_test)
        
        # Calculate metrics for each target column
        print("\nModel Performance Metrics:")
        print("-" * 50)
        for i, col in enumerate(target_columns):
            print(f"\n{col}:")
            print(f"Accuracy: {accuracy_score(y_test[:, i], y_pred[:, i]):.3f}")
            print(f"F1 Score: {f1_score(y_test[:, i], y_pred[:, i], average='binary'):.3f}")
            print(f"Precision: {precision_score(y_test[:, i], y_pred[:, i], average='binary'):.3f}")
            print(f"Recall: {recall_score(y_test[:, i], y_pred[:, i], average='binary'):.3f}")
        
        # Save the damage analysis for the model to use
        model.set_damage_analysis(movement_patterns, attack_patterns)
    else:
        print("Failed to train model!")

if __name__ == "__main__":
    train_model() 