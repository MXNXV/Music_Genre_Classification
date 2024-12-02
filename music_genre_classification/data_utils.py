import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer
import json
from tqdm import tqdm
import librosa
import warnings
warnings.filterwarnings('ignore')

class MusicGenreDataset(Dataset):
    """Dataset class for multi-modal music genre classification"""
    
    def __init__(self, audio_dir, lyrics_file, transform=None, max_length=512):
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.max_length = max_length
        
        # Load lyrics data
        self.data = pd.read_json(lyrics_file)
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Genre mapping
        self.genre_map = {
            'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
            'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7,
            'reggae': 8, 'rock': 9
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and preprocess audio
        audio_path = self.audio_dir / f"{row['track_id']}.wav"
        mel_spec = self._process_audio(audio_path)
        
        # Process lyrics
        lyrics = row['lyrics']
        input_ids, attention_mask = self._process_lyrics(lyrics)
        
        # Get label
        label = self.genre_map[row['genre']]
        
        return {
            'audio': mel_spec,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'track_id': row['track_id']
        }
    
    def _process_audio(self, audio_path):
        """Process audio file to mel-spectrogram"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Create mel-spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )
            
            mel_spec = mel_transform(waveform)
            
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Apply any additional transforms
            if self.transform:
                mel_spec = self.transform(mel_spec)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {str(e)}")
            return torch.zeros((1, 128, 128))  # Return zero tensor as fallback
    
    def _process_lyrics(self, lyrics):
        """Process lyrics using BERT tokenizer"""
        encoding = self.tokenizer(
            lyrics,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

class DataAugmentation:
    """Audio data augmentation techniques"""
    
    @staticmethod
    def time_stretch(mel_spec, rate=1.2):
        return torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0),
            scale_factor=(1, rate),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    @staticmethod
    def pitch_shift(mel_spec, bins_shift=2):
        return torch.roll(mel_spec, shifts=bins_shift, dims=1)
    
    @staticmethod
    def add_noise(mel_spec, noise_level=0.005):
        noise = torch.randn_like(mel_spec) * noise_level
        return mel_spec + noise

class EvaluationMetrics:
    """Class for computing and visualizing evaluation metrics"""
    
    def __init__(self, model, test_loader, device, genre_labels):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.genre_labels = genre_labels
        
    def compute_metrics(self):
        """Compute all evaluation metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move inputs to device
                audio = batch['audio'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions
                outputs = self.model(audio, input_ids, attention_mask)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute confusion matrix
        self.conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Get classification report
        self.class_report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.genre_labels,
            output_dict=True
        )
        
        return self.conf_matrix, self.class_report
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.conf_matrix,
            xticklabels=self.genre_labels,
            yticklabels=self.genre_labels,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_performance_by_genre(self, save_path=None):
        """Plot performance metrics by genre"""
        metrics = pd.DataFrame({
            'Precision': [self.class_report[genre]['precision'] for genre in self.genre_labels],
            'Recall': [self.class_report[genre]['recall'] for genre in self.genre_labels],
            'F1-Score': [self.class_report[genre]['f1-score'] for genre in self.genre_labels]
        }, index=self.genre_labels)
        
        metrics.plot(kind='bar', figsize=(12, 6))
        plt.title('Performance Metrics by Genre')
        plt.xlabel('Genre')
        plt.ylabel('Score')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """Plot training and validation metrics over epochs"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def prepare_data_loaders(audio_dir, lyrics_file, batch_size=32, train_split=0.8, val_split=0.1):
    """Prepare DataLoaders for training, validation, and testing"""
    
    # Create dataset
    dataset = MusicGenreDataset(audio_dir, lyrics_file)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    # Set up data paths
    AUDIO_DIR = "path/to/audio/files"
    LYRICS_FILE = "path/to/lyrics.json"
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        AUDIO_DIR,
        LYRICS_FILE,
        batch_size=32
    )
    
    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicGenreClassifier().to(device)
    
    # Train model and get history
    history = train_model(model, train_loader, val_loader)
    
    # Initialize evaluation metrics
    genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                   'jazz', 'metal', 'pop', 'reggae', 'rock']
    evaluator = EvaluationMetrics(model, test_loader, device, genre_labels)
    
    # Compute and visualize metrics
    conf_matrix, class_report = evaluator.compute_metrics()
    evaluator.plot_confusion_matrix(save_path='confusion_matrix.png')
    evaluator.plot_performance_by_genre(save_path='genre_performance.png')
    evaluator.plot_training_history(history, save_path='training_history.png')
    
    # Save classification report
    with open('classification_report.json', 'w') as f:
        json.dump(class_report, f, indent=4)
