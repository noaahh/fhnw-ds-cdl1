# Model Overview

## Explored Models

### Logistic Regression

### Convolutional Neural Network (CNN)
- **Paper**: [Convolutional Networks for Images, Speech, and Time-Series](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e26cc4a1c717653f323715d751c8dea7461aa105)
- **Implementation**: [cnn.py](/src/models/cnn.py)
- **Model Config**: [cnn.yaml](/configs/model/cnn.yaml)

1-dimensional convolutional neural networks (1D CNNs) are suitable for classifying human sensor activity because they can efficiently capture local patterns in time-series data, which is common in sensor readings. By applying convolutional filters along the time dimension, 1D CNNs can detect and learn important features such as spikes, trends, and periodicities that correspond to different activities. Additionally, 1D CNNs have fewer parameters compared to RNNs (i.e. LSTMs), making them faster to train and less prone to overfitting, which is beneficial for large-scale sensor data.

### Long Short-Term Memory Model (LSTM)
- **Paper**: [Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
- **Implementation**: [lstm.py](/src/models/lstm.py)
- **Model Config**: [lstm.yaml](/configs/model/lstm.yaml)

A simple LSTM is well-suited for Sensor Activity Recognition due to its ability to capture long-term dependencies and temporal patterns in sequential data, which is what sensor readings are. LSTMs maintain and update a memory cell that allows them to remember important information over long sequences, making them adept at modeling the temporal dynamics of sensor data. Additionally, LSTMs are relatively straightforward to implement and have proven robustness in various time-series prediction tasks, providing a reliable and efficient option for recognizing activities from sensor inputs.

### Extended Long Short-Term Memory (xLSTM)
- **Paper**: [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- **Implementation**: [x_lstm.py](/src/models/x_lstm.py)
- **Model Config**: [x_lstm.yaml](/configs/model/x_lstm.yaml)

In the 90s, LSTMs introduced gating and the constant error carousel, forming the basis for the first Large Language Models (LLMs). The xLSTM architecture explores scaling LSTMs to billions of parameters by introducing exponential gating with appropriate normalization and stabilization techniques, and modifying the LSTM memory structure into sLSTM (with scalar memory and update) and mLSTM (with matrix memory and covariance update rule). These modifications, integrated into residual block backbones and stacked into xLSTM architectures, enhance performance and scalability, making xLSTMs competitive with state-of-the-art Transformers and State Space Models.

Even though the xLSTM architecture was introduced in the context of Large Language Modelling, we propose that the xLSTM model's advanced memory structures and gating mechanisms make it also suitable for Sensor Activity Recognition, as they enable efficient handling of long-term dependencies and complex temporal patterns in sensor data. The improved stability and scalability of xLSTM can lead to more accurate and robust activity recognition, even in scenarios with large-scale and high-dimensional sensor inputs, which is what the data at hand consists of.

### Deep Residual Bidirectional LSTM
- **Paper**: [Deep Residual Bidir-LSTM for Human Activity Recognition Using Wearable Sensors](https://arxiv.org/pdf/1708.08989v2)
- **Implementation**: [deep_res_bidir_lstm.py](/src/models/deep_res_bidir_lstm.py)
- **Model Config**: [deep_res_bidir_lstm.yaml](/configs/model/lsdeep_res_bidir_lstm.yaml)

For this model's implementation we found inspiration [in this paper](https://arxiv.org/pdf/1708.08989v2)). This research group found success with the incorporation of two LSTM layers running in parallel but in opposite time directions. One processes the sequence forward, while the other processes it backward, allowing the model to capture context from both past and future states.

### Transformer
- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Implementation**: [transformer.py](/src/models/transformer.py)
- **Model Config**: [transformer.yaml](/configs/model/transformer.yaml)

We propose that the self-attention mechanism, parallel processing capabilities, scalability, flexibility, and robust performance of Transformers as seen in numerous established applications can make a powerful choice for Sensor Activity Recognition.

## Comparison between Classical and Deep Learning Models