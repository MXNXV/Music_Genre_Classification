from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

class EvaluationMetrics:
    def __init__(self, model, test_loader, device, genre_labels):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.genre_labels = genre_labels

    def compute_metrics(self):
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                audio = batch['audio'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                outputs = self.model(audio, input_ids, attention_mask)
                _, predictions = torch.max(outputs, 1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.genre_labels))))
        class_report = classification_report(all_labels, all_preds, target_names=self.genre_labels, output_dict=True)

        return conf_matrix, class_report

    def plot_confusion_matrix(self, save_path=None):
        conf_matrix, _ = self.compute_metrics()
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=self.genre_labels, yticklabels=self.genre_labels, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_performance_by_genre(self, save_path=None):
        _, class_report = self.compute_metrics()
        f1_scores = [class_report[label]['f1-score'] for label in self.genre_labels]

        plt.figure(figsize=(10, 5))
        plt.bar(self.genre_labels, f1_scores)
        plt.title("F1-Score by Genre")
        plt.xlabel("Genre")
        plt.ylabel("F1-Score")
        plt.ylim(0, 1)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_training_history(self, training_history, save_path=None):
        epochs = range(1, len(training_history['train_loss']) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, training_history['train_loss'], label='Train Loss')
        plt.plot(epochs, training_history['val_loss'], label='Validation Loss')
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, training_history['train_acc'], label='Train Accuracy')
        plt.plot(epochs, training_history['val_acc'], label='Validation Accuracy')
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()


class MusicGenreDataset(Dataset):
    def __init__(self, audio_dir, lyrics_file, transform=None, max_length=512):
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.max_length = max_length
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load lyrics data
        self.data = pd.read_json(lyrics_file)
        
        # Map genres to numerical labels
        self.genre_map = {genre: idx for idx, genre in enumerate(self.data['genre'].unique())}
        self.data['genre'] = self.data['genre'].map(self.genre_map)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
    
    # Ensure track_id is a string
        track_id = str(row['track_id'])
    
    # Extract the numeric part of track_id
        numeric_part = int(track_id.split('_')[-1])
        processed_numeric_part = ''.join(
        [char for i, char in enumerate(str(numeric_part)) if i == 0 or char != str(numeric_part)[i-1]])
    
    # Determine the folder (genre) based on the numeric range
        genre_folder = (int(processed_numeric_part) - 1) // 100  # Integer division to find the folder (0 for 1-100, 1 for 101-200, etc.)
        
    # Construct the correct file name
        file_name = f"{genre_folder}_{processed_numeric_part}.wav"
        audio_path = self.audio_dir / str(genre_folder) / file_name
    
    # Process audio
        mel_spec = self._process_audio(audio_path)
    
    # Process lyrics
        lyrics = row['lyrics']
        input_ids, attention_mask = self._process_lyrics(lyrics)
    
    # Get label
        label = row['genre']
    
        return {
        'audio': mel_spec,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label,
        'track_id': row['track_id']
    }


    
    def _process_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=2048,
                hop_length=512,
                n_mels=64
            )
            mel_spec = mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-9)
            mel_spec = torch.nn.functional.interpolate(
                mel_spec.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            return mel_spec
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return torch.zeros((1, 64, 64))
    
    def _process_lyrics(self, lyrics):
        encoding = self.tokenizer(
            lyrics,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

def prepare_data_loaders(audio_dir, lyrics_file, batch_size=32):
    dataset = MusicGenreDataset(audio_dir, lyrics_file)
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=dataset.data['genre'], random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=dataset.data['genre'].iloc[temp_idx], random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_dataset(lyrics_file, audio_dir):
    """
    Analyzes the dataset for missing or mismatched audio files and visualizes genre distribution.

    Args:
        lyrics_file (str): Path to the lyrics JSON file.
        audio_dir (str): Path to the audio directory.
    """
    dataset = pd.read_json(lyrics_file)
    print("\nDataset Analysis:")
    print("Total samples:", len(dataset))
    
    # Check genre distribution
    genre_counts = dataset['genre'].value_counts()
    print("\nSamples per genre:")
    print(genre_counts)
    
    # Check for missing or mismatched audio files
    print("\nChecking audio files...")
    missing_files = []
    mismatched_files = []
    for _, row in dataset.iterrows():
        genre = str(row['genre'])  # Ensure genre is string for path consistency
        track_id = row['track_id']
        expected_path = Path(audio_dir) / genre / f"{track_id}.wav"
        
        # Check if the expected file exists
        if not expected_path.exists():
            # Check if a different file exists in the same genre folder
            genre_folder = Path(audio_dir) / genre
            actual_files = {file.name for file in genre_folder.glob("*.wav")}
            if not any(f"{track_id}.wav" in fname for fname in actual_files):
                missing_files.append(str(expected_path))
            else:
                mismatched_files.append((track_id, genre_folder))
    
    if missing_files:
        print(f"\nMissing audio files ({len(missing_files)}):")
        for file in missing_files:
            print(file)
    else:
        print("No missing files detected.")
    
    if mismatched_files:
        print(f"\nMismatched audio files ({len(mismatched_files)}):")
        for track_id, folder in mismatched_files:
            print(f"Track ID {track_id} in folder {folder}")
    else:
        print("No mismatched files detected.")
    
    # Plot genre distribution
    plt.figure(figsize=(10, 5))
    genre_counts.plot(kind='bar')
    plt.title('Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('genre_distribution.png')
    plt.close()
    print("\nGenre distribution plot saved as 'genre_distribution.png'.")

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_data_split(lyrics_file, audio_dir, output_dir='output'):
    """
    Analyzes the data distribution across train, validation, and test splits.

    Args:
        lyrics_file (str): Path to the lyrics JSON file.
        audio_dir (str): Path to the audio directory.
        output_dir (str): Path to save analysis results.
    """
    # Load dataset
    dataset = pd.read_json(lyrics_file)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\nAnalyzing dataset splits...")
    genre_counts = dataset['genre'].value_counts()

    # Split dataset into train, val, test
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=dataset['genre'], random_state=42)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=dataset['genre'].iloc[temp_idx], random_state=42)

    train_set = dataset.iloc[train_idx]
    val_set = dataset.iloc[val_idx]
    test_set = dataset.iloc[test_idx]

    # Genre distribution in each split
    train_dist = train_set['genre'].value_counts()
    val_dist = val_set['genre'].value_counts()
    test_dist = test_set['genre'].value_counts()

    print("\nGenre distribution:")
    print("Train Set:")
    print(train_dist)
    print("\nValidation Set:")
    print(val_dist)
    print("\nTest Set:")
    print(test_dist)

    # Plot distributions
    plt.figure(figsize=(12, 6))
    train_dist.plot(kind='bar', color='blue', alpha=0.6, label='Train')
    val_dist.plot(kind='bar', color='orange', alpha=0.6, label='Validation')
    test_dist.plot(kind='bar', color='green', alpha=0.6, label='Test')
    plt.title("Genre Distribution Across Splits")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "split_distribution.png")
    plt.close()

    print(f"Split distribution plot saved to {output_path / 'split_distribution.png'}")

    # Save split information
    train_set.to_json(output_path / "train_split.json", orient="records", indent=4)
    val_set.to_json(output_path / "val_split.json", orient="records", indent=4)
    test_set.to_json(output_path / "test_split.json", orient="records", indent=4)

    print(f"Data splits saved to {output_dir}.")

    return {
        "train_dist": train_dist.to_dict(),
        "val_dist": val_dist.to_dict(),
        "test_dist": test_dist.to_dict(),
    }

