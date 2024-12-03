import torch
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import gc

from classifier import LightMusicGenreClassifier

from data_utils import prepare_data_loaders, EvaluationMetrics, analyze_data_split


# Inside main()



def setup_logging():
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
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration
    CONFIG = {
        'audio_dir': r'data\audio',
        'lyrics_file': r'data\lyrics.json',
        'batch_size': 8,            
        'num_epochs': 20,           
        'learning_rate': 0.00005,   
        'fusion_type': 'concatenate',
        'num_genres': 10,
        'output_dir': r'output'
    }
    
    # Create output directory
    Path(CONFIG['output_dir']).mkdir(exist_ok=True)
    
    # Save configuration
    with open(Path(CONFIG['output_dir']) / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    try:
        # Analyze data first
        analyze_data_split(
    lyrics_file=CONFIG['lyrics_file'],
    audio_dir=CONFIG['audio_dir'],
    output_dir=CONFIG['output_dir']
)
        # logger.info("Analyzing dataset...")
        # genre_dist = analyze_data_split()
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Data loading
        logger.info("Loading data...")
        train_loader, val_loader, test_loader = prepare_data_loaders(
            CONFIG['audio_dir'],
            CONFIG['lyrics_file'],
            batch_size=CONFIG['batch_size']
        )
        logger.info("Data loaded successfully")
        
        # Initialize model
        logger.info("Initializing model...")
        model = LightMusicGenreClassifier(
            num_genres=CONFIG['num_genres'],
            fusion_type=CONFIG['fusion_type']
        )
        
        # Move model to device
        model = model.to(device)
        model.criterion = model.criterion.to(device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
            
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
            for batch in train_pbar:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                audio = batch['audio'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(audio, input_ids, attention_mask)
                loss = model.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                train_loss += loss.item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
                
                del audio, input_ids, attention_mask, labels, outputs
                gc.collect()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            val_pbar = tqdm(val_loader, desc='Validation')
            with torch.no_grad():
                for batch in val_pbar:
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
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
                    
                    del audio, input_ids, attention_mask, labels, outputs
                    gc.collect()
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_acc = 100. * val_correct / val_total
            
            training_history['train_loss'].append(epoch_train_loss)
            training_history['val_loss'].append(epoch_val_loss)
            training_history['train_acc'].append(epoch_train_acc)
            training_history['val_acc'].append(epoch_val_acc)
            
            print(f'\nEpoch {epoch+1} Results:')
            print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
            
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr:.6f}')
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), 
                         Path(CONFIG['output_dir']) / 'best_model.pth')
                print("Saved new best model")
            
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Evaluation
        logger.info("Starting evaluation...")
        genre_labels = [str(i) for i in range(10)]
        evaluator = EvaluationMetrics(model, test_loader, device, genre_labels)
        
        # Model predictions analysis
        logger.info("\nAnalyzing model predictions...")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                audio = batch['audio'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                
                outputs = model(audio, input_ids, attention_mask)
                _, predictions = outputs.max(1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        print("\nPredictions Analysis:")
        print("-" * 40)
        print("Unique predicted classes:", set(all_preds))
        print("Unique true classes:", set(all_labels))
        
        for pred, true in zip(all_preds, all_labels):
            print(f"Predicted: {pred}, True: {true}")
        
        # Compute and save metrics
        conf_matrix, class_report = evaluator.compute_metrics()
        evaluator.plot_confusion_matrix(
            save_path=Path(CONFIG['output_dir']) / 'confusion_matrix.png')
        evaluator.plot_performance_by_genre(
            save_path=Path(CONFIG['output_dir']) / 'genre_performance.png')
        evaluator.plot_training_history(
            training_history, 
            save_path=Path(CONFIG['output_dir']) / 'training_history.png')
        
        with open(Path(CONFIG['output_dir']) / 'classification_report.json', 'w') as f:
            json.dump(class_report, f, indent=4)
        
        logger.info("Training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()