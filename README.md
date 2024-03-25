## Variational Auto-Encoder (VAE) Applications

This project uses a Variational Auto-Encoder architecture

## What is a VAE?


VAEs are generative models that learn a compressed latent representation of the data. They consist of an encoder, which converts data samples into latent vectors, and a decoder, which reconstructs the data samples from the latent vectors.

![VAE Architecture](/images/architecture.png)

## VAE Applications
- Dimensionality Reduction
- Data Generation
- Denoising
- Anomaly Detection
- Image Inpainting

## What is Anomaly Detection?
Anomaly detection involves identifying samples that are significantly different from typical data. VAEs can be used for anomaly detection by detecting samples that result in a large reconstruction error, indicating they are anomalies.



## Detecting Anomalies using VAE
- Data Preparation: Gather and preprocess data, including train/test split.
- Model Training: Build a VAE and train it on the training set.
- Anomaly Detection: Pass test samples to the VAE and record the reconstruction loss for each.
- Identification: Identify test samples with a reconstruction loss higher than a set criterion as anomalies.



## Dataset
The project uses a public dataset from Kaggle (https://www.kaggle.com/boltzmannbrain/nab), which contains one-dimensional timeseries data. Feature engineering is applied to make the data multi-dimensional, improving model accuracy.


## Fraud Detection:

Fraud detection is the process of identifying fraudulent activities in financial transactions. It involves detecting patterns or anomalies that indicate fraudulent behavior, such as unusual spending patterns or unauthorized access.

## Detecting Frauds using VAE:

To detect frauds using VAE, you can follow these steps:

- Data Preprocessing: Gather and preprocess data, including separating normal transactions from fraudulent transactions.
- Model Training: Train a VAE on the normal transactions to learn the distribution of normal behavior.
- Anomaly Detection: Pass both normal and fraudulent transactions through the trained VAE. Calculate the reconstruction error for each transaction.

- Identification: Identify transactions with high reconstruction error as potential frauds.

- Evaluation: Evaluate the performance of the VAE-based fraud detection system using metrics like precision, recall, and F1-score.

## Dataset
The project uses a public dataset from Kaggle (https://www.kaggle.com/code/hone5com/fraud-detection-with-variational-autoencoder/input)

# Requirements
- sklearn
- torch
- matplotlib
- numpy
- pandas
