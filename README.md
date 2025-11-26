# Advanced Time Series Forecasting with Deep Learning and Explainability (LSTM)

This project implements an end-to-end multivariate time series forecasting pipeline using a sequence model (LSTM) and model explainability (SHAP).

## Dataset

- Programmatically generated multivariate time series
- 5 input features (`sensor_1` … `sensor_5`) mimicking industrial sensors
- 2000+ time steps with:
  - linear trend
  - multiple seasonalities (periods 30, 50, 70)
  - Gaussian noise
- Target variable: one-step-ahead forecast of `sensor_1`

The dataset satisfies the assignment requirement of a complex, real-world-mimicking series with seasonality, trend and noise.

## Model

- Many-to-one LSTM forecaster (TensorFlow/Keras)
- Input: 30-step sliding window of 5 features
- Architecture:
  - 1–2 stacked LSTM layers (32–64 units)
  - Final Dense layer with 1 output neuron
- Loss: MSE
- Optimizer: Adam

## Hyperparameter Tuning

Simple grid search over:

- `num_layers`: {1, 2}
- `hidden_size`: {32, 64}
- `learning_rate`: {0.001, 0.0003}
- `batch_size`: {32, 64}

Best configuration is selected using validation MAE, then retrained on train + validation data with early stopping.

## Metrics

The model is evaluated on a time-based test split using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## Explainability (SHAP)

SHAP DeepExplainer is applied to the trained LSTM:

- Global feature importance = mean |SHAP| over all time steps
- Time-step importance = mean |SHAP| over all features within the input window

This reveals which sensors and which parts of the recent history the model relies on most for its forecasts.

## How to Run

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib shap
python main_lstm_time_series_project.py
