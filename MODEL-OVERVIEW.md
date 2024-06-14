# Model Overview

## Explored Models

### Logistic Regression

### Convolutional Neural Network (CNN)

### Deep Residual Bidirectional LSTM

### Long Short-Term Memory Model (LSTM)

### Extended Long Short-Term Memory (xLSTM)
**Paper**: [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
**Implementation**: [x_lstm.py](/src/models/x_lstm.py)
**Model Config**: [x_lstm.yaml](/configs/model/x_lstm.yaml)

In the 90s, LSTMs introduced gating and the constant error carousel, forming the basis for the first Large Language Models (LLMs). The xLSTM architecture explores scaling LSTMs to billions of parameters by introducing exponential gating with appropriate normalization and stabilization techniques, and modifying the LSTM memory structure into sLSTM (with scalar memory and update) and mLSTM (with matrix memory and covariance update rule). These modifications, integrated into residual block backbones and stacked into xLSTM architectures, enhance performance and scalability, making xLSTMs competitive with state-of-the-art Transformers and State Space Models.

Even though the xLSTM architecture was introduced in the context of Large Language Modelling, we propose that the xLSTM model's advanced memory structures and gating mechanisms make it also suitable for Sensor Activity Recognition, as they enable efficient handling of long-term dependencies and complex temporal patterns in sensor data. The improved stability and scalability of xLSTM can lead to more accurate and robust activity recognition, even in scenarios with large-scale and high-dimensional sensor inputs, which is what the data at hand consists of.

### Transformer
**Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
**Implementation**: [transformer.py](/src/models/transformer.py)
**Model Config**: [transformer.yaml](/configs/model/transformer.yaml)

We propose that the self-attention mechanism, parallel processing capabilities, scalability, flexibility, and robust performance of Transformers as seen in numerous established applications can make a powerful choice for Sensor Activity Recognition.

## Comparison between Classical and Deep Learning Models