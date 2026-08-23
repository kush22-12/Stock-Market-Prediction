# Stock Market Prediction using Hybrid LSTM-GRU

A deep learning based stock-market forecasting project that uses a **Hybrid LSTM-GRU architecture** to predict the next 4 closing prices from the previous 30 market candles.

## Project Overview

This project develops a multi-step time-series forecasting pipeline for stock market data. Historical OHLCV data is transformed through technical-indicator based feature engineering, normalized using a StandardScaler, converted into sliding-window sequences, and passed to a Hybrid LSTM-GRU neural network.

### Forecasting Setup

- **Input window:** Previous 30 candles
- **Prediction horizon:** Next 4 candles
- **Input features:** 21 engineered market features
- **Target:** Closing price
- **Architecture:** Hybrid LSTM + GRU
- **LSTM layers:** 2
- **GRU layers:** 2
- **Hidden size:** 128
- **Dropout:** 0.2

## Pipeline

```text
Raw OHLCV Data
      ↓
Feature Engineering
      ↓
Technical Indicators
      ↓
Sliding Window
(30 historical candles)
      ↓
StandardScaler
(fit on training data)
      ↓
Tensor Sequences
      ↓
Hybrid LSTM + GRU
      ↓
4-Step Forecast
      ↓
Predicted Close Prices

