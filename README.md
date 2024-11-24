# Music_Genre_Classification
# Music Genre Classification

**Deep Learning for Automated Music Genre Classification Using Audio and Lyrical Features**  
_Final Project 1 - Fall 2024_  

**Contributors**:  
- Manav Mandal  
- Hrushikesh Attarde  
- Jayanand Hiremath  

---

## Research Question

How can a multi-modal deep learning approach combining audio and lyrical features improve the accuracy of automated music genre classification compared to single-modality methods?

---

## Background

Music genre classification plays a key role in music information retrieval, enabling applications such as recommendation systems, playlist generation, and music organization. Traditional methods rely on handcrafted audio features, but advancements in deep learning have demonstrated the potential for automatically learning features from raw audio data. 

Key insights include:  
- Convolutional Neural Networks (CNNs) have shown state-of-the-art results using spectrograms for audio-only classification.  
- Textual content, such as lyrics, remains an underexplored modality for enhancing genre classification accuracy.  

This project bridges the gap by developing a **multi-modal approach** that leverages both audio and lyrical features, aiming to achieve superior classification performance.

---

## Data

1. **Audio Data**:  
   - **Dataset**: GTZAN Genre Collection  
   - **Description**: 1000 audio tracks (30 seconds each) distributed across 10 genres.  
   - **Format**: `.wav` files, sampled at 22050 Hz.  

2. **Lyrical Data**:  
   - **Dataset**: musiXmatch, part of the Million Song Dataset.  
   - Matched lyrics for the GTZAN tracks. Manual collection will ensure full lyrical coverage.

---

## Methodology

### 1. Audio Preprocessing  
- Convert raw audio files into **mel-spectrograms**.  
- Extract **Mel-frequency cepstral coefficients (MFCCs)** using the `librosa` library.

### 2. Lyrical Preprocessing  
- Tokenize and encode lyrics using a **pre-trained BERT model**.

### 3. Model Architecture  
The model combines audio and lyrical features using:  
- **CNN** for audio feature extraction.  
- **BERT-based model** for lyrical analysis.  
- **Fusion Layer** to merge audio and lyrical embeddings.  
- **Classification Layer** for predicting genres.

### 4. Training and Evaluation  
- **5-fold cross-validation** will be used to train and evaluate the model.  
- Performance will be compared against audio-only and lyrics-only models.

### 5. Analysis  
- Performance analysis across different genres.  
- Visualize feature importance using **attention visualization techniques**.

---

## Expected Outcomes

- Enhanced classification accuracy by combining audio and lyrical features.  
- Insights into the complementary nature of audio and lyrical modalities in genre identification.

---

## References

1. [GTZAN Genre Collection](https://www.tensorflow.org/datasets/catalog/gtzan)  
2. [musiXmatch Dataset](http://millionsongdataset.com/musixmatch/)  
3. [Deep Learning Music Genre Classification](https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning)  
4. [Music Genre Classification Research](https://transactions.ismir.net/articles/10.5334/tismir.10)  

For detailed citations, see the project proposal.

---

## Citations

- http://millionsongdataset.com  
- https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning  
- https://www.tensorflow.org/datasets/catalog/gtzan  
- https://huggingface.co/datasets/marsyas/gtzan

---

## Project Structure

project_directory/
    ├── music_genre_classifier.py     # First file (model architecture)
    ├── music_genre_data_utils.py     # Second file (data loading and evaluation)
    ├── main.py                       # Third file (main execution script)
    ├── output/                       # Created automatically
    └── data/
        ├── audio/                    # Your GTZAN audio files
        └── lyrics.json               # Your lyrics data

---

## Updating Configuration in `main.py`

Ensure the configuration paths in `main.py` are correctly set:

```python
CONFIG = {
    'audio_dir': 'data/audio',        # Path to audio data directory
    'lyrics_file': 'data/lyrics.json', # Path to lyrics file
    ...
}


## Run training

python main.py
