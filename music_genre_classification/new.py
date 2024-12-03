import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_dataset(lyrics_file, audio_dir):
    """
    Analyzes the dataset for missing audio files and visualizes genre distribution.
    
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
    
    # Check for missing audio files
    print("\nChecking audio files...")
    missing_files = []
    for _, row in dataset.iterrows():
        genre = str(row['genre'])  # Ensure genre is string for path consistency
        track_id = row['track_id']
        expected_path = Path(audio_dir) / genre / f"{track_id}.wav"
        
        if not expected_path.exists():
            missing_files.append(str(expected_path))
    
    if missing_files:
        print(f"\nMissing audio files ({len(missing_files)}):")
        for file in missing_files:
            print(file)
    else:
        print("All audio files are present.")
    
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
