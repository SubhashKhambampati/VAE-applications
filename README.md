## Variational Auto-Encoder (VAE) Applications

This project uses a Variational Auto-Encoder architecture

## What is a VAE?


VAEs are generative models that learn a compressed latent representation of the data. They consist of an encoder, which converts data samples into latent vectors, and a decoder, which reconstructs the data samples from the latent vectors.

![VAE Architecture](/imgs/architecture.png)

## What is Anomaly Detection?
Anomaly detection involves identifying samples that are significantly different from typical data. VAEs can be used for anomaly detection by detecting samples that result in a large reconstruction error, indicating they are anomalies.



## Detecting Anomalies using VAE
- Data Preparation: Gather and preprocess data, including train/test split.
- Model Training: Build a VAE and train it on the training set.
- Anomaly Detection: Pass test samples to the VAE and record the reconstruction loss for each.
- Identification: Identify test samples with a reconstruction loss higher than a set criterion as anomalies.



## Dataset
The project uses a public dataset from Kaggle (https://www.kaggle.com/boltzmannbrain/nab), which contains one-dimensional timeseries data. Feature engineering is applied to make the data multi-dimensional, improving model accuracy.
