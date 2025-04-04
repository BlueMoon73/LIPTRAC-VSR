# LIP-TRAC: Visual Speech Recognition for Enhanced Communication

## Overview

LIP-TRAC is a research project focused on developing an accessible and efficient visual speech recognition (VSR) system. It aims to address the challenges faced by individuals with hearing impairments by transcribing speech from lip movements. This project prioritizes both accuracy and real-time performance, to maximize its usability in practical, everyday scenarios. It utilizes the power of artificial intelligence to address current gaps in the area of visual speech recognition.

## Project Goals

*   To develop a lightweight and efficient model for real-time lipreading.
*   To balance transcription accuracy with low latency to enable practical use.
*   To offer a cost-effective and accessible solution compared to existing assistive technologies.

## Key Features

*   **Lightweight CRNN Architecture:** Employs a convolutional recurrent neural network (CRNN) architecture designed for efficient computation.
*   **Connectionist Temporal Classification (CTC) Loss:** Utilizes CTC loss to handle the unaligned nature of lip movements and audio transcriptions.
*   **In-Video Normalization:** A technique to improve signal detection, and reduce the amount of resources needed to store training data.
*   **Raspberry Pi Deployment:** The code can be deployed on a Raspberry Pi, a low cost and widely available platform.
*   **Haar Cascade Face Detection**: Utilizes Haar Cascade classifiers to automate lip region cropping.

## Methodology

The LIP-TRAC pipeline can be broken down into four main stages:

1.  **Data Curation:**
    *   The BBC LRS2 dataset was used for training.
    *   Frames are pre-processed by cropping the lip region using computer vision (HAAR Cascade Classifiers).
2.  **Normalization and Encoding:**
    *   The visual data is normalized to account for lighting variations, improving robustness.
    *   Transcript data is encoded for use in machine learning.
3.  **Training:**
    *   A CRNN model is trained, using transfer learning and batch normalizations.
    *   The model was trained to maximize for accuracy and minimize for inference time.
    *   A custom learning rate scheduler is used for more efficient and robust training.
4.  **Evaluation & Deployment:**
    *   The trained model is tested on a dedicated data set.
    *  To ensure real time capability, the test is conducted by a computer, to quantify the inference time.
    *   Results are evaluated against key metrics, to ensure that it is optimized for real world performance.

## Model Architecture

The model is a hybrid architecture that consists of both 3D convolutional and recurrent components. It follows a Convolutional Recurrent Neural Network (CRNN).

*   **3D Convolution**: 3D convolution layers that extract both spatial and temporal features from the video.
*   **GRU Layers**: Gated Recurrent Unit layers to model the temporal context of lip movements.
*   **CTC Loss**: A method for connecting the model to a text output.

## Future Work


*   **Improved Robustness:** To add to its use in the real world, try improving the model to deal with noisy data, such as different lighting conditions, different video angles, and backgrounds.
*   **Multilingual Capability**: Develop the model to perform in languages beyond English.
*   **N-Gram Implementation**: To predict sequences of words utilizing context, in a similar fashion to "Autocorrect" on your phone. 

## Contact 
Portfolio: [Get in touch!](https://monishsaravana.com/)
