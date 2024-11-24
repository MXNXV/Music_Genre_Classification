import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import torchaudio
import numpy as np

class AudioCNN(nn.Module):
    def __init__(self, num_channels=1, num_features=128):
        super(AudioCNN, self).__init__()
        
        # CNN layers for processing mel-spectrograms
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Final dense layer
        self.fc = nn.Linear(512 * 4 * 4, num_features)
        
    def forward(self, x):
        # Apply CNN layers with residual connections
        x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool(F.relu(self.bn3(self.conv3(x2))))
        x4 = self.pool(F.relu(self.bn4(self.conv4(x3))))
        
        # Flatten and apply dense layer
        x = x4.view(-1, 512 * 4 * 4)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LyricBERT(nn.Module):
    def __init__(self, num_features=128):
        super(LyricBERT, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Dense layer to match feature dimensions with audio
        self.fc = nn.Linear(768, num_features)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and dense layer
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

class MultiModalFusion(nn.Module):
    def __init__(self, num_features=128, num_genres=10, fusion_type='attention'):
        super(MultiModalFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Multi-head attention fusion
            self.attention = nn.MultiheadAttention(num_features, num_heads=4)
            self.norm1 = nn.LayerNorm(num_features)
            self.norm2 = nn.LayerNorm(num_features)
        
        # Final classification layers
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, num_genres)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, audio_features, lyric_features):
        if self.fusion_type == 'attention':
            # Reshape for attention
            audio_features = audio_features.unsqueeze(0)
            lyric_features = lyric_features.unsqueeze(0)
            
            # Apply cross-modal attention
            attn_output, _ = self.attention(audio_features, lyric_features, lyric_features)
            
            # Add residual connections and layer normalization
            fusion = self.norm1(audio_features + attn_output)
            fusion = fusion.squeeze(0)
            
        elif self.fusion_type == 'concatenate':
            # Simple concatenation
            fusion = torch.cat((audio_features, lyric_features), dim=1)
            
        elif self.fusion_type == 'gated':
            # Gated fusion mechanism
            gate = torch.sigmoid(audio_features * lyric_features)
            fusion = gate * audio_features + (1 - gate) * lyric_features
        
        # Final classification
        x = F.relu(self.fc1(fusion))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MusicGenreClassifier(nn.Module):
    def __init__(self, num_genres=10, fusion_type='attention'):
        super(MusicGenreClassifier, self).__init__()
        
        # Audio and lyric feature extractors
        self.audio_cnn = AudioCNN()
        self.lyric_bert = LyricBERT()
        
        # Fusion module
        self.fusion = MultiModalFusion(fusion_type=fusion_type)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, audio_input, input_ids, attention_mask):
        # Extract features from both modalities
        audio_features = self.audio_cnn(audio_input)
        lyric_features = self.lyric_bert(input_ids, attention_mask)
        
        # Fuse features and classify
        output = self.fusion(audio_features, lyric_features)
        return output

# Data preprocessing functions
def preprocess_audio(audio_path, sample_rate=22050, duration=30):
    """Convert audio to mel-spectrogram."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Create mel-spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )(waveform)
    
    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)
    return mel_spec

def preprocess_lyrics(lyrics, tokenizer, max_length=512):
    """Tokenize lyrics using BERT tokenizer."""
    encoding = tokenizer(
        lyrics,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (audio, lyrics, labels) in enumerate(train_loader):
            audio = audio.to(device)
            input_ids = lyrics['input_ids'].to(device)
            attention_mask = lyrics['attention_mask'].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio, input_ids, attention_mask)
            loss = model.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, lyrics, labels in val_loader:
                audio = audio.to(device)
                input_ids = lyrics['input_ids'].to(device)
                attention_mask = lyrics['attention_mask'].to(device)
                labels = labels.to(device)
                
                outputs = model(audio, input_ids, attention_mask)
                loss = model.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        # Print epoch statistics
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

