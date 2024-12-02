import os
import requests
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_gtzan_sample():
    """
    Downloads a small sample of GTZAN dataset from reliable mirrors
    """
    logger = setup_logging()
    
    # Create directories
    base_dir = Path('data')
    audio_dir = base_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample tracks from each genre (using reliable hosting)
    sample_tracks = {
        'blues': ['https://example.com/blues1.wav', 'https://example.com/blues2.wav'],
        'classical': ['https://example.com/classical1.wav', 'https://example.com/classical2.wav'],
        # Add more genre samples
    }
    
    def download_file(url, filepath):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False

    logger.info("Starting sample audio downloads...")
    for genre, urls in sample_tracks.items():
        genre_dir = audio_dir / genre
        genre_dir.mkdir(exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, url in enumerate(urls):
                filepath = genre_dir / f"track{i+1}.wav"
                futures.append(executor.submit(download_file, url, filepath))

def create_sample_lyrics():
    """
    Creates a sample lyrics dataset matching the audio files
    """
    logger = setup_logging()
    
    # Sample lyrics data
    sample_lyrics = {
        'blues1': {
            'track_id': 'blues1',
            'lyrics': "Sample blues lyrics...",
            'genre': 'blues'
        },
        'classical1': {
            'track_id': 'classical1',
            'lyrics': "Sample classical lyrics...",
            'genre': 'classical'
        },
        # Add more samples
    }
    
    # Save to JSON
    lyrics_file = Path('data') / 'lyrics.json'
    with open(lyrics_file, 'w') as f:
        json.dump(list(sample_lyrics.values()), f, indent=4)
    
    logger.info(f"Created sample lyrics file at {lyrics_file}")

def download_from_huggingface():
    """
    Alternative method to download GTZAN from Hugging Face
    """
    logger = setup_logging()
    
    try:
        from datasets import load_dataset
        
        logger.info("Downloading GTZAN dataset from Hugging Face...")
        dataset = load_dataset("marsyas/gtzan", split="train")
        
        # Create directories
        audio_dir = Path('data/audio')
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and save audio files
        for item in tqdm(dataset, desc="Downloading audio files"):
            genre = item['genre']
            genre_dir = audio_dir / genre
            genre_dir.mkdir(exist_ok=True)
            
            # Save audio file
            audio_path = genre_dir / f"{item['track_id']}.wav"
            if not audio_path.exists():
                sf.write(audio_path, item['audio']['array'], item['audio']['sampling_rate'])
        
        logger.info("Download completed successfully")
        
    except Exception as e:
        logger.error(f"Error downloading from Hugging Face: {str(e)}")
        logger.info("Please try manual download from: https://huggingface.co/datasets/marsyas/gtzan")

def create_dummy_dataset(num_samples=10):
    """
    Creates a dummy dataset for testing
    """
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
        genre_dir = audio_dir / genre
        genre_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            # Generate dummy audio (white noise)
            sr = 22050
            duration = 30  # seconds
            audio = np.random.randn(sr * duration)
            
            # Save audio file
            track_id = f"{genre}_{i+1}"
            audio_path = genre_dir / f"{track_id}.wav"
            sf.write(audio_path, audio, sr)
            
            # Create dummy lyrics
            dummy_data.append({
                'track_id': track_id,
                'lyrics': f"Sample lyrics for {genre} song {i+1}...",
                'genre': genre
            })
    
    # Save dummy lyrics
    with open(base_dir / 'lyrics.json', 'w') as f:
        json.dump(dummy_data, f, indent=4)
    
    logger.info(f"Created dummy dataset with {len(dummy_data)} samples")
    return dummy_data

if __name__ == "__main__":
    logger = setup_logging()
    
    print("\nMusic Genre Classification Dataset Setup")
    print("----------------------------------------")
    print("1. Download sample GTZAN dataset")
    print("2. Download full GTZAN dataset from Hugging Face")
    print("3. Create dummy dataset for testing")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        download_gtzan_sample()
        create_sample_lyrics()
    elif choice == '2':
        download_from_huggingface()
    elif choice == '3':
        num_samples = int(input("Enter number of samples per genre (default 10): ") or 10)
        create_dummy_dataset(num_samples)
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice!")