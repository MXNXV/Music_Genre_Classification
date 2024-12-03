import os
import numpy as np
import json
import torch
import torchaudio
from pathlib import Path
import logging
from tqdm import tqdm
from datasets import load_dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_dummy_dataset(num_samples=10):
    """Creates a dummy dataset for testing"""
    logger = setup_logging()
    
    # Create directories
    base_dir = Path('data')
    audio_dir = base_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy audio files
    logger.info("Generating dummy audio files...")
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop']
    dummy_data = []
    
    for genre in genres:
        # Create genre directory
        genre_dir = audio_dir / genre
        genre_dir.mkdir(exist_ok=True, parents=True)
        
        for i in range(num_samples):
            # Generate dummy audio (white noise)
            sr = 22050
            duration = 3  # seconds (reduced for testing)
            num_samples_audio = sr * duration
            
            # Create random noise as a tensor
            audio = torch.randn(1, num_samples_audio)
            
            # Create proper track id and path
            track_id = f"{genre}_{i+1}"
            audio_path = genre_dir / f"{track_id}.wav"
            
            # Save using torchaudio
            torchaudio.save(
                str(audio_path),
                audio,
                sr,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            # Create dummy lyrics
            dummy_data.append({
                'track_id': track_id,
                'lyrics': f"Sample lyrics for {genre} song {i+1}...",
                'genre': genre
            })
    
    # Save dummy lyrics
    lyrics_file = base_dir / 'lyrics.json'
    with open(lyrics_file, 'w') as f:
        json.dump(dummy_data, f, indent=4)
    
    logger.info(f"Created dummy dataset with {len(dummy_data)} samples")
    return dummy_data

def download_gtzan_subset():
    """Downloads a small subset of the GTZAN dataset"""
    logger = setup_logging()
    
    try:
        logger.info("Downloading GTZAN dataset...")
        dataset = load_dataset("marsyas/gtzan", split="train[:1000]", trust_remote_code=True)
        
        # Create directories with r-strings
        base_dir = Path(r'data')
        audio_dir = Path(r'data\audio')
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metadata for lyrics.json
        metadata = []
        
        # Process dataset
        for idx, item in enumerate(tqdm(dataset, desc="Processing audio files")):
            try:
                genre = item['genre']
                
                # Create genre directory with r-string
                genre_dir = Path(rf'data\audio\{genre}')
                genre_dir.mkdir(exist_ok=True, parents=True)
                
                # Get audio data
                audio_array = np.array(item['audio']['array'])
                sr = item['audio']['sampling_rate']
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
                
                # Create file name with r-string
                track_id = f"{genre}_{idx+1}"
                audio_path = genre_dir / f"{track_id}.wav"
                
                # Save audio file
                torchaudio.save(
                    str(audio_path),
                    audio_tensor,
                    sr,
                    encoding='PCM_S',
                    bits_per_sample=16
                )
                
                # Add metadata
                metadata.append({
                    'track_id': track_id,
                    'lyrics': f"Instrumental music - {genre} genre",
                    'genre': genre
                })
                
            except Exception as e:
                logger.error(f"Error processing {genre} file {idx+1}: {str(e)}")
                continue
        
        # Save lyrics data with r-string
        lyrics_path = Path(r'data\lyrics.json')
        with open(lyrics_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Print final statistics
        files = list(Path(r'data\audio').rglob('*.wav'))
        logger.info(f"Dataset creation complete. Total files: {len(files)}")
        
        # Print genre distribution
        genre_counts = {}
        for file in files:
            genre = file.parent.name
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        logger.info("Files per genre:")
        for genre, count in genre_counts.items():
            logger.info(f"{genre}: {count} files")
            
    except Exception as e:
        logger.error(f"Error downloading GTZAN dataset: {str(e)}")
        logger.info("Please ensure you have the 'datasets' package installed: pip install datasets")
        
        
if __name__ == "__main__":
    logger = setup_logging()
    
    print("\nMusic Genre Classification Dataset Setup")
    print("----------------------------------------")
    print("1. Download sample GTZAN dataset")
    print("2. Download small subset of GTZAN dataset")
    print("3. Create dummy dataset for testing")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            logger.error("Sample GTZAN dataset download not implemented")
        elif choice == '2':
            logger.info("Starting GTZAN subset download...")
            download_gtzan_subset()
        elif choice == '3':
            num_samples = int(input("Enter number of samples per genre (default 10): ") or 10)
            create_dummy_dataset(num_samples)
        elif choice == '4':
            print("Exiting...")
        else:
            print("Invalid choice!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)