"""
Time Series LSTM Model for Football Momentum Prediction

This module implements a Long Short-Term Memory (LSTM) neural network for capturing
temporal patterns and momentum in football team performance. The model analyzes
sequences of recent matches to predict future outcomes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os
from os import path


class FootballLSTM(nn.Module):
    """LSTM model for football match prediction based on temporal sequences"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super(FootballLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 3),  # 3 outputs: Home Win, Draw, Away Win
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        output = self.fc(last_output)
        return output


class FootballSequenceDataset(Dataset):
    """Dataset class for football match sequences"""

    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMPredictor:
    """LSTM-based predictor for football matches using temporal sequences"""

    def __init__(self, sequence_length=5, hidden_size=64, num_layers=2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_match_features(self, match_sequence, team_name):
        """
        Extract features from a sequence of matches for a specific team

        Args:
            match_sequence: DataFrame of recent matches
            team_name: Name of the team to analyze

        Returns:
            numpy.ndarray: Feature vector for the sequence
        """
        features = []

        for _, match in match_sequence.iterrows():
            match_features = []

            # Basic match result (from team's perspective)
            if match['HomeTeam'] == team_name:
                # Team is playing at home
                match_features.extend([
                    1,  # Home flag
                    0,  # Away flag
                    match.get('HomeShots', 0),
                    match.get('AwayShots', 0),
                    match.get('HomeShotsOnTarget', 0),
                    match.get('AwayShotsOnTarget', 0),
                    match.get('HomeCorners', 0),
                    match.get('AwayCorners', 0),
                    match.get('HomeFouls', 0),
                    match.get('AwayFouls', 0),
                    match.get('HomeYellowCards', 0),
                    match.get('AwayYellowCards', 0),
                    match.get('HomeRedCards', 0),
                    match.get('AwayRedCards', 0)
                ])

                # Result encoding
                if match['FullTimeResult'] == 'H':
                    result = [1, 0, 0]  # Win
                elif match['FullTimeResult'] == 'D':
                    result = [0, 1, 0]  # Draw
                else:
                    result = [0, 0, 1]  # Loss

            else:
                # Team is playing away
                match_features.extend([
                    0,  # Home flag
                    1,  # Away flag
                    match.get('AwayShots', 0),
                    match.get('HomeShots', 0),
                    match.get('AwayShotsOnTarget', 0),
                    match.get('HomeShotsOnTarget', 0),
                    match.get('AwayCorners', 0),
                    match.get('HomeCorners', 0),
                    match.get('AwayFouls', 0),
                    match.get('HomeFouls', 0),
                    match.get('AwayYellowCards', 0),
                    match.get('HomeYellowCards', 0),
                    match.get('AwayRedCards', 0),
                    match.get('HomeRedCards', 0)
                ])

                # Result encoding (flipped for away perspective)
                if match['FullTimeResult'] == 'A':
                    result = [1, 0, 0]  # Win
                elif match['FullTimeResult'] == 'D':
                    result = [0, 1, 0]  # Draw
                else:
                    result = [0, 0, 1]  # Loss

            match_features.extend(result)
            features.append(match_features)

        return np.array(features)

    def prepare_sequence_data(self, df, sequence_length=None):
        """
        Prepare time series sequences for LSTM training

        Args:
            df: DataFrame with historical match data
            sequence_length: Number of matches to include in each sequence

        Returns:
            tuple: (sequences, labels) as numpy arrays
        """
        if sequence_length is None:
            sequence_length = self.sequence_length

        sequences = []
        labels = []

        # Get unique teams
        teams = df['HomeTeam'].unique()

        for team in teams:
            # Get all matches for this team
            team_matches = df[
                (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
            ].sort_values('MatchDate').reset_index(drop=True)

            # Skip if team doesn't have enough matches
            if len(team_matches) < sequence_length + 1:
                continue

            # Create sequences
            for i in range(len(team_matches) - sequence_length):
                # Sequence of past matches
                seq_matches = team_matches.iloc[i:i+sequence_length]
                seq_features = self.extract_match_features(seq_matches, team)

                # Target match (the next one)
                target_match = team_matches.iloc[i+sequence_length]

                # Get label for target match
                if target_match['HomeTeam'] == team:
                    if target_match['FullTimeResult'] == 'H':
                        label = 0  # Home win
                    elif target_match['FullTimeResult'] == 'D':
                        label = 1  # Draw
                    else:
                        label = 2  # Away win
                else:
                    if target_match['FullTimeResult'] == 'A':
                        label = 0  # Away win (from team's perspective)
                    elif target_match['FullTimeResult'] == 'D':
                        label = 1  # Draw
                    else:
                        label = 2  # Home win (from team's perspective = loss)

                sequences.append(seq_features.flatten())  # Flatten the sequence
                labels.append(label)

        return np.array(sequences), np.array(labels)

    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model

        Args:
            X_train: Training sequences
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Reshape for LSTM: (batch_size, sequence_length, input_size)
        # Derive input_size from data to avoid hardcoding mismatch
        sequence_length = self.sequence_length
        input_size = X_train.shape[1] // sequence_length
        self.input_size = input_size  # persist for predict_proba and save/load

        X_train_reshaped = X_train_scaled.reshape(-1, sequence_length, input_size)

        # Create dataset and dataloader
        dataset = FootballSequenceDataset(X_train_reshaped, y_train)

        # Split into train/validation
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = FootballLSTM(input_size, self.hidden_size, self.num_layers).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_accuracy = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = 100 * train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)

                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = 100 * val_correct / val_total

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
                      f'Val Acc: {val_accuracy:.2f}%')

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/lstm_best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        # Load best model
        if path.exists('models/lstm_best_model.pth'):
            self.model.load_state_dict(torch.load('models/lstm_best_model.pth'))

        self.is_trained = True
        print(f'✅ LSTM model trained successfully! Best validation accuracy: {best_val_accuracy:.2f}%')

    def predict_proba(self, sequences):
        """
        Predict probabilities for sequences

        Args:
            sequences: Input sequences

        Returns:
            numpy.ndarray: Prediction probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()

        # Scale and reshape
        sequences_scaled = self.scaler.transform(sequences)
        input_size = getattr(self, 'input_size', sequences.shape[1] // self.sequence_length)
        sequence_length = self.sequence_length
        sequences_reshaped = sequences_scaled.reshape(-1, sequence_length, input_size)

        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences_reshaped).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def predict_match(self, home_team, away_team, historical_df, recent_matches=5):
        """
        Predict a specific match using recent team performance

        Args:
            home_team: Home team name
            away_team: Away team name
            historical_df: DataFrame with historical match data
            recent_matches: Number of recent matches to use for each team

        Returns:
            dict: Prediction results
        """
        try:
            # Get recent matches for home team
            home_recent = historical_df[
                (historical_df['HomeTeam'] == home_team) | (historical_df['AwayTeam'] == home_team)
            ].sort_values('MatchDate', ascending=False).head(recent_matches)

            # Get recent matches for away team
            away_recent = historical_df[
                (historical_df['HomeTeam'] == away_team) | (historical_df['AwayTeam'] == away_team)
            ].sort_values('MatchDate', ascending=False).head(recent_matches)

            if len(home_recent) < recent_matches or len(away_recent) < recent_matches:
                return {
                    'error': f'Insufficient historical data for {home_team} or {away_team}',
                    'HomeWinProb': 0.33,
                    'DrawProb': 0.34,
                    'AwayWinProb': 0.33
                }

            # For now, use home team's recent form as primary predictor
            # In a more sophisticated approach, we'd combine both teams' sequences
            home_sequence = self.extract_match_features(home_recent.sort_values('MatchDate'), home_team)

            if len(home_sequence) == 0:
                return {
                    'error': f'No sequence data available for {home_team}',
                    'HomeWinProb': 0.33,
                    'DrawProb': 0.34,
                    'AwayWinProb': 0.33
                }

            # Flatten and predict
            sequence_flat = home_sequence.flatten().reshape(1, -1)
            probabilities = self.predict_proba(sequence_flat)

            return {
                'HomeWinProb': float(probabilities[0][0]),
                'DrawProb': float(probabilities[0][1]),
                'AwayWinProb': float(probabilities[0][2])
            }

        except Exception as e:
            return {
                'error': f'LSTM prediction failed: {str(e)}',
                'HomeWinProb': 0.33,
                'DrawProb': 0.34,
                'AwayWinProb': 0.33
            }

    def save_model(self, filepath):
        """Save the LSTM predictor to disk"""
        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained,
            'input_size': getattr(self, 'input_size', 17)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath):
        """Load an LSTM predictor from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls(
            sequence_length=model_data['sequence_length'],
            hidden_size=model_data['hidden_size'],
            num_layers=model_data['num_layers'],
            learning_rate=model_data['learning_rate']
        )

        predictor.scaler = model_data['scaler']
        predictor.is_trained = model_data['is_trained']

        if model_data['model_state_dict']:
            input_size = model_data.get('input_size', 17)
            predictor.input_size = input_size
            predictor.model = FootballLSTM(input_size, predictor.hidden_size, predictor.num_layers)
            predictor.model.load_state_dict(model_data['model_state_dict'])
            predictor.model.to(predictor.device)

        return predictor


def create_lstm_predictor(sequence_length=5, hidden_size=64, num_layers=2):
    """Factory function to create an LSTM predictor"""
    return LSTMPredictor(sequence_length, hidden_size, num_layers)


def train_lstm_model(historical_df, sequence_length=5, epochs=50):
    """
    Convenience function to train an LSTM model

    Args:
        historical_df: DataFrame with historical match data
        sequence_length: Number of matches in each sequence
        epochs: Number of training epochs

    Returns:
        LSTMPredictor: Trained LSTM predictor
    """
    predictor = LSTMPredictor(sequence_length=sequence_length)

    # Prepare data
    X, y = predictor.prepare_sequence_data(historical_df, sequence_length)

    if len(X) == 0:
        raise ValueError("No valid sequences found in the data")

    print(f"Prepared {len(X)} sequences for training")

    # Train model
    predictor.train_model(X, y, epochs=epochs)

    return predictor


DEFAULT_MODEL_PATH = path.join(path.dirname(__file__), 'lstm_predictor.pkl')

def predict_match_lstm(home_team, away_team, historical_df, model_path=None):
    """
    Convenience function for single match prediction

    Args:
        home_team: Home team name
        away_team: Away team name
        historical_df: DataFrame with historical match data
        model_path: Path to saved model (optional, defaults to models/lstm_predictor.pkl)

    Returns:
        dict: Prediction results
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if path.exists(model_path):
        predictor = LSTMPredictor.load_model(model_path)
    else:
        # Train on historical data and save for future use
        print(f"No saved LSTM model found at {model_path}. Training on historical data...")
        predictor = train_lstm_model(historical_df, sequence_length=5, epochs=30)
        predictor.save_model(model_path)

    return predictor.predict_match(home_team, away_team, historical_df)


if __name__ == "__main__":
    print("LSTM Predictor for Football Momentum")
    print("=" * 40)
    print("This module implements time series LSTM for capturing team momentum.")
    print("Use train_lstm_model() to train on historical data.")
    print("Use predict_match_lstm() for match predictions.")