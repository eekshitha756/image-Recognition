# Face Recognition using PCA and ANN

This project implements a complete face recognition system using Principal Component Analysis (PCA) for feature extraction and an Artificial Neural Network (ANN) for classification.

## Project Structure

- `dataset/`: Contains the face images for training/testing.
- `src/`: Contains individual modules for the project.
  - `load_dataset.py`: Reads and preprocesses image files.
  - `pca_feature_extraction.py`: Performs PCA computation.
  - `eigenfaces.py`: Projects images and generates signatures.
  - `ann_classifier.py`: Trains and tests the Backpropagation ANN.
  - `train.py`: Pipeline for training models and saving them to disk.
  - `test.py`: Pipeline for testing new images, including Imposter Detection.
- `results/`: Output directory where models and accuracy graphs are saved.
- `main.py`: Entry point for evaluating the system across different 'k' values.
- `requirements.txt`: Python dependencies.

## Setup and Execution

1. **Install Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Application:**
   ```bash
   python main.py
   ```
   
   Running `main.py` will:
   - Load the dataset from `dataset/faces`.
   - Iterate through defined $k$ values (Number of Eigenvectors kept).
   - Train an ANN and test it with a 60/40 train/test split.
   - Plot and save the `accuracy_graph.png`.
   - Display the Top Eigenfaces visually.
   - Run an Imposter Detection test on an un-enrolled face image.

## Imposter Detection

The ANN output probabilities are used as a confidence score. If the maximum confidence probability falls below a set threshold (e.g., 50%), the testing module identifies the face as a "Not Enrolled Person (Imposter)". This is demonstrated at the end of `main.py`.
