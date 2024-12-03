import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class LightAudioCNN(nn.Module):
    def __init__(self, num_channels=1, num_features=64):
        super(LightAudioCNN, self).__init__()
        
        # Very simple CNN
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, num_features, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)

class LightLyricProcessor(nn.Module):
    def __init__(self, num_features=64):
        super(LightLyricProcessor, self).__init__()
        
        # Use smaller BERT model
        self.bert = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        
        # Freeze BERT parameters to reduce memory usage
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.fc = nn.Linear(128, num_features)  # BERT-tiny has 128 hidden size
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # No gradient computation for BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token
        
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

class LightMusicGenreClassifier(nn.Module):
    def __init__(self, num_genres=10, fusion_type='concatenate'):
        super(LightMusicGenreClassifier, self).__init__()
        
        # Feature extractors
        self.audio_cnn = LightAudioCNN(num_features=64)
        self.lyric_processor = LightLyricProcessor()
        
        self.fusion_type = fusion_type
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_genres)
        )
        
        # Initialize criterion with default weights
        self.criterion = nn.CrossEntropyLoss()
    
    def set_class_weights(self, class_counts):
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        weights = weights / weights.sum() * len(weights)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, audio_input, input_ids, attention_mask):
        # Extract features
        audio_features = self.audio_cnn(audio_input)
        lyric_features = self.lyric_processor(input_ids, attention_mask)
        
        # Combine features
        if self.fusion_type == 'concatenate':
            combined = torch.cat((audio_features, lyric_features), dim=1)
        elif self.fusion_type == 'sum':
            combined = audio_features + lyric_features
        else:
            combined = torch.cat((audio_features, lyric_features), dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output