import torch
from pathlib import Path
import json
import logging
from datetime import datetime

from classifier import MusicGenreClassifier
from data_utils import prepare_data_loaders, EvaluationMetrics

def setup_logging():
    """Set up logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration
    CONFIG = {
        'audio_dir': r'data\audio',
        'lyrics_file': r'data\lyrics.json',
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'fusion_type': 'attention',
        'num_genres': 5,
        'output_dir': 'output'
    }
    
    # Create output directory
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save configuration
    with open(Path(CONFIG['output_dir']) / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    logger.info("Starting music genre classification training")
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Prepare data loaders
        logger.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = prepare_data_loaders(
            CONFIG['audio_dir'],
            CONFIG['lyrics_file'],
            batch_size=CONFIG['batch_size']
        )
        logger.info("Data loaders prepared successfully")
        
        # 2. Initialize model
        logger.info(f"Initializing model with {CONFIG['fusion_type']} fusion...")
        model = MusicGenreClassifier(
            num_genres=CONFIG['num_genres'],
            fusion_type=CONFIG['fusion_type']
        ).to(device)
        
        # 3. Train model
        logger.info("Starting training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        best_val_loss = float('inf')
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(CONFIG['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Move data to device
                audio = batch['audio'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(audio, input_ids, attention_mask)
                loss = model.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    audio = batch['audio'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(audio, input_ids, attention_mask)
                    loss = model.criterion(outputs, labels)
                    
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_loss += loss.item()
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_acc = 100. * val_correct / val_total
            
            # Update training history
            training_history['train_loss'].append(epoch_train_loss)
            training_history['val_loss'].append(epoch_val_loss)
            training_history['train_acc'].append(epoch_train_acc)
            training_history['val_acc'].append(epoch_val_acc)
            
            logger.info(f'Epoch: {epoch+1}/{CONFIG["num_epochs"]}')
            logger.info(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            logger.info(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), 
                         Path(CONFIG['output_dir']) / 'best_model.pth')
                logger.info("Saved new best model")
        
        # 4. Evaluate model
        logger.info("Starting evaluation...")
        genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop']
        
        evaluator = EvaluationMetrics(model, test_loader, device, genre_labels)
        
        # Compute and save metrics
        conf_matrix, class_report = evaluator.compute_metrics()
        evaluator.plot_confusion_matrix(
            save_path=Path(CONFIG['output_dir']) / 'confusion_matrix.png')
        evaluator.plot_performance_by_genre(
            save_path=Path(CONFIG['output_dir']) / 'genre_performance.png')
        evaluator.plot_training_history(
            training_history, 
            save_path=Path(CONFIG['output_dir']) / 'training_history.png')
        
        # Save classification report
        with open(Path(CONFIG['output_dir']) / 'classification_report.json', 'w') as f:
            json.dump(class_report, f, indent=4)
        
        logger.info("Training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()