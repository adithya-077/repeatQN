This repository contains the implementation of a Question Similarity model based on GPT architecture, combined with an additional DistilBERT for feature extraction. The model is designed to predict the similarity between pairs of input questions, which is useful in NLP applications like duplicate question detector.

Model Architecture:
Feature Extraction: The model utilizes the [CLS] token representation from DistilBERT for both input sequences (pair of questions).
Feature Transformation: An MLP block processes the concatenated [CLS] embeddings of both questions, followed by a series of transformations and residual connections.
Prediction: A series of fully connected layers (with activation functions) outputs the final similarity prediction between the two input questions.

Requirements:
torch (PyTorch)
transformers (Hugging Face's Transformer library)
numpy
tqdm (for progress bars)
