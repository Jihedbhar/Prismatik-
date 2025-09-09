# models_Sales_forecasting.py - UPDATED WITH FIXED TIME SERIES AND TRANSFORMER MODELS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Add TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Input, MultiHeadAttention, LayerNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Keras models will be skipped.")

# Add time series models imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Time series models will be skipped.")

try:
    from pmdarima import auto_arima

    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("pmdarima not available. Auto-ARIMA models will be skipped.")

try:
    from prophet import Prophet
    import logging

    logging.getLogger('prophet').setLevel(logging.WARNING)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Prophet models will be skipped.")


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.best_params = None
        self.best_score = None
        self.feature_importance = None

    @abstractmethod
    def create_model(self, trial: optuna.Trial, n_features: int) -> Any:
        pass

    @abstractmethod
    def train_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        pass

    @abstractmethod
    def predict(self, model: Any, X_test: pd.DataFrame) -> np.ndarray:
        pass

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
        if not valid_mask.any():
            return {
                'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'),
                'r2': -float('inf'), 'mape': float('inf')
            }

        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        return {
            'mse': mean_squared_error(y_true_valid, y_pred_valid),
            'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
            'mae': mean_absolute_error(y_true_valid, y_pred_valid),
            'r2': r2_score(y_true_valid, y_pred_valid),
            'mape': np.mean(np.abs((y_true_valid - y_pred_valid) / np.maximum(y_true_valid, 1e-8))) * 100
        }

    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        try:
            model = self.create_model(trial, X_train.shape[1])
            trained_model = self.train_model(model, X_train, y_train)
            y_pred = self.predict(trained_model, X_val)

            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return float('inf')

            return mean_squared_error(y_val, y_pred)
        except Exception:
            return float('inf')

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.config.optuna_trials,
            timeout=self.config.optuna_timeout
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials)
        }



class BaselineModel(BaseModel):
    """Baseline model that predicts the mean"""

    def create_model(self, trial: optuna.Trial, n_features: int) -> float:
        return 0.0

    def train_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        return y_train.mean()

    def predict(self, model: float, X_test: pd.DataFrame) -> np.ndarray:
        return np.full(len(X_test), model)

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        mean_pred = y_train.mean()
        baseline_pred = np.full(len(y_val), mean_pred)
        mse = mean_squared_error(y_val, baseline_pred)

        return {
            'best_params': {'baseline_mean': mean_pred},
            'best_score': mse,
            'n_trials': 1
        }


class RandomForestModel(BaseModel):
    """Random Forest model"""

    def create_model(self, trial: optuna.Trial, n_features: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            random_state=self.config.random_state,
            n_jobs=-1
        )

    def train_model(self, model: RandomForestRegressor, X_train: pd.DataFrame,
                    y_train: pd.Series) -> RandomForestRegressor:
        model.fit(X_train, y_train)
        self.feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        return model

    def predict(self, model: RandomForestRegressor, X_test: pd.DataFrame) -> np.ndarray:
        return model.predict(X_test)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model"""

    def create_model(self, trial: optuna.Trial, n_features: int) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            random_state=self.config.random_state
        )

    def train_model(self, model: GradientBoostingRegressor, X_train: pd.DataFrame,
                    y_train: pd.Series) -> GradientBoostingRegressor:
        model.fit(X_train, y_train)
        self.feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        return model

    def predict(self, model: GradientBoostingRegressor, X_test: pd.DataFrame) -> np.ndarray:
        return model.predict(X_test)


class RidgeModel(BaseModel):
    """Ridge Regression model"""

    def create_model(self, trial: optuna.Trial, n_features: int) -> Ridge:
        return Ridge(
            alpha=trial.suggest_float('alpha', 0.01, 10.0),
            random_state=self.config.random_state
        )

    def train_model(self, model: Ridge, X_train: pd.DataFrame, y_train: pd.Series) -> Ridge:
        model.fit(X_train, y_train)
        self.feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
        return model

    def predict(self, model: Ridge, X_test: pd.DataFrame) -> np.ndarray:
        return model.predict(X_test)


class XGBoostModel(BaseModel):
    """XGBoost model"""

    def create_model(self, trial: optuna.Trial, n_features: int) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=0
        )

    def train_model(self, model: xgb.XGBRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        model.fit(X_train, y_train)
        self.feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        return model

    def predict(self, model: xgb.XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
        return model.predict(X_test)


class LightGBMModel(BaseModel):
    """LightGBM model"""

    def create_model(self, trial: optuna.Trial, n_features: int) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1
        )

    def train_model(self, model: lgb.LGBMRegressor, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMRegressor:
        model.fit(X_train, y_train)
        self.feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        return model

    def predict(self, model: lgb.LGBMRegressor, X_test: pd.DataFrame) -> np.ndarray:
        return model.predict(X_test)


# FIXED TIME SERIES MODELS
if STATSMODELS_AVAILABLE:
    class ARIMAModel(BaseModel):
        """FIXED ARIMA time series model"""

        def __init__(self, config):
            super().__init__(config)
            self.is_fitted = False

        def create_model(self, trial: optuna.Trial, n_features: int) -> tuple:
            # Conservative parameter space for stability
            p = trial.suggest_int('p', 0, 3)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 3)
            return (p, d, q)

        def train_model(self, model: tuple, X_train: pd.DataFrame, y_train: pd.Series):
            p, d, q = model
            try:
                # Ensure we have enough data points
                if len(y_train) < 10:
                    print(f"Warning: Only {len(y_train)} data points for ARIMA")
                    return self._fallback_model(y_train)

                # Clean the data
                y_clean = y_train.dropna()
                if len(y_clean) == 0:
                    return self._fallback_model(y_train)

                # Fit ARIMA model
                arima_model = ARIMA(y_clean, order=(p, d, q))
                fitted_model = arima_model.fit()
                self.is_fitted = True
                return fitted_model

            except Exception as e:
                print(f"ARIMA fitting failed: {e}, using fallback")
                return self._fallback_model(y_train)

        def _fallback_model(self, y_train):
            """Simple fallback when ARIMA fails"""

            class FallbackModel:
                def __init__(self, mean_value):
                    self.mean_value = mean_value
                    self.fittedvalues = pd.Series([mean_value] * len(y_train))

                def forecast(self, steps):
                    return np.full(steps, self.mean_value)

            return FallbackModel(y_train.mean())

        def predict(self, model, X_test: pd.DataFrame) -> np.ndarray:
            try:
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(steps=len(X_test))
                    # Ensure positive predictions for sales
                    forecast = np.maximum(forecast, 0)
                    return forecast
                else:
                    return np.full(len(X_test), model.mean_value)
            except Exception as e:
                print(f"ARIMA prediction failed: {e}")
                return np.full(len(X_test), 1.0)


    class ExponentialSmoothingModel(BaseModel):
        """FIXED Exponential Smoothing model"""

        def create_model(self, trial: optuna.Trial, n_features: int) -> dict:
            return {
                'trend': trial.suggest_categorical('trend', [None, 'add']),
                'seasonal': None,  # Disable seasonal for stability
                'seasonal_periods': None
            }

        def train_model(self, model: dict, X_train: pd.DataFrame, y_train: pd.Series):
            try:
                # Clean data
                y_clean = y_train.dropna()
                if len(y_clean) < 3:
                    return self._create_fallback(y_train.mean())

                # Try exponential smoothing
                es_model = ExponentialSmoothing(
                    y_clean,
                    trend=model['trend'],
                    seasonal=model['seasonal']
                )
                fitted_model = es_model.fit(optimized=True)
                return fitted_model

            except Exception as e:
                print(f"Exponential Smoothing failed: {e}")
                return self._create_fallback(y_train.mean())

        def _create_fallback(self, mean_val):
            class SimpleFallback:
                def __init__(self, value):
                    self.value = value

                def forecast(self, steps):
                    return np.full(steps, self.value)

            return SimpleFallback(mean_val)

        def predict(self, model, X_test: pd.DataFrame) -> np.ndarray:
            try:
                forecast = model.forecast(len(X_test))
                return np.maximum(forecast, 0)
            except:
                return np.full(len(X_test), getattr(model, 'value', 1.0))

if STATSMODELS_AVAILABLE and PMDARIMA_AVAILABLE:
    class AutoARIMAModel(BaseModel):
        """FIXED Auto-ARIMA model using pmdarima"""

        def create_model(self, trial: optuna.Trial, n_features: int) -> dict:
            return {
                'seasonal': trial.suggest_categorical('seasonal', [True, False]),
                'm': trial.suggest_int('m', 7, 12) if trial.params.get('seasonal', False) else 1,
                'max_p': trial.suggest_int('max_p', 0, 3),
                'max_q': trial.suggest_int('max_q', 0, 3),
                'max_order': 6
            }

        def train_model(self, model: dict, X_train: pd.DataFrame, y_train: pd.Series):
            try:
                y_clean = y_train.dropna()
                if len(y_clean) < 20:  # Need more data for seasonal models
                    model['seasonal'] = False
                    model['m'] = 1

                fitted_model = auto_arima(
                    y_clean,
                    seasonal=model['seasonal'],
                    m=model.get('m', 1),
                    max_p=model['max_p'],
                    max_q=model['max_q'],
                    max_order=model['max_order'],
                    suppress_warnings=True,
                    error_action='ignore',
                    stepwise=True,
                    n_fits=20  # Limit computation
                )
                return fitted_model

            except Exception as e:
                print(f"Auto-ARIMA failed: {e}")

                # Final fallback
                class SimpleMean:
                    def __init__(self, value):
                        self.value = value

                    def predict(self, n_periods):
                        return np.full(n_periods, self.value)

                return SimpleMean(y_train.mean())

        def predict(self, model, X_test: pd.DataFrame) -> np.ndarray:
            try:
                if hasattr(model, 'predict'):
                    forecast = model.predict(n_periods=len(X_test))
                else:
                    forecast = np.full(len(X_test), model.value)
                return np.maximum(forecast, 0)
            except Exception as e:
                print(f"Auto-ARIMA prediction failed: {e}")
                return np.full(len(X_test), 1.0)

if PROPHET_AVAILABLE:
    class ProphetModel(BaseModel):
        """FIXED Facebook Prophet model"""

        def create_model(self, trial: optuna.Trial, n_features: int) -> dict:
            return {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10),
                'yearly_seasonality': False,  # Disable for short series
                'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
                'daily_seasonality': False
            }

        def train_model(self, model: dict, X_train: pd.DataFrame, y_train: pd.Series):
            try:
                # Create Prophet dataframe with proper date column
                if 'date' in X_train.columns:
                    dates = pd.to_datetime(X_train['date'])
                else:
                    # Generate dates if not available
                    dates = pd.date_range(start='2023-01-01', periods=len(y_train), freq='D')

                prophet_df = pd.DataFrame({
                    'ds': dates,
                    'y': y_train.values
                })

                # Remove any invalid data
                prophet_df = prophet_df.dropna()
                if len(prophet_df) < 10:
                    return self._create_fallback(y_train.mean())

                # Initialize Prophet with conservative settings
                prophet_model = Prophet(
                    changepoint_prior_scale=model['changepoint_prior_scale'],
                    seasonality_prior_scale=model['seasonality_prior_scale'],
                    yearly_seasonality=model['yearly_seasonality'],
                    weekly_seasonality=model['weekly_seasonality'],
                    daily_seasonality=model['daily_seasonality'],
                    interval_width=0.8,
                    uncertainty_samples=100  # Reduce for speed
                )

                # Fit with error handling
                prophet_model.fit(prophet_df)
                return prophet_model

            except Exception as e:
                print(f"Prophet training failed: {e}")
                return self._create_fallback(y_train.mean())

        def _create_fallback(self, mean_val):
            class ProphetFallback:
                def __init__(self, value):
                    self.value = value

                def predict(self, future_df):
                    result = pd.DataFrame({
                        'yhat': [self.value] * len(future_df)
                    })
                    return result

            return ProphetFallback(mean_val)

        def predict(self, model, X_test: pd.DataFrame) -> np.ndarray:
            try:
                # Create future dataframe
                if 'date' in X_test.columns:
                    future_dates = pd.to_datetime(X_test['date'])
                else:
                    # Generate future dates
                    future_dates = pd.date_range(start='2024-01-01', periods=len(X_test), freq='D')

                future_df = pd.DataFrame({'ds': future_dates})
                forecast = model.predict(future_df)

                if hasattr(forecast, 'yhat'):
                    predictions = forecast['yhat'].values
                else:
                    predictions = np.full(len(X_test), model.value)

                return np.maximum(predictions, 0)

            except Exception as e:
                print(f"Prophet prediction failed: {e}")
                return np.full(len(X_test), 1.0)

# FIXED TRANSFORMER AND DEEP LEARNING MODELS
if TENSORFLOW_AVAILABLE:
    def create_sequences(X, y, n_lags):
        """Create sequences for time series models - ROBUST VERSION"""
        if len(X) <= n_lags:
            return np.array([]), np.array([])

        X_seq, y_seq = [], []
        for i in range(n_lags, len(X)):
            X_seq.append(X[i - n_lags:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)


    class KerasLSTMModel(BaseModel):
        """FIXED LSTM model using Keras/TensorFlow"""

        def __init__(self, config):
            super().__init__(config)
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            self.n_lags = 10

        def create_model(self, trial: optuna.Trial, n_features: int) -> Sequential:
            # Adaptive lag selection based on data size
            max_lags = min(20, n_features, 15)
            self.n_lags = trial.suggest_int('n_lags', 5, max_lags)

            units = trial.suggest_int('units', 32, 128)
            n_layers = trial.suggest_int('n_layers', 1, 2)
            dropout = trial.suggest_float('dropout', 0.0, 0.3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            model = Sequential()

            if n_layers == 1:
                model.add(LSTM(units, activation="tanh", input_shape=(self.n_lags, n_features)))
            else:
                model.add(LSTM(units, activation="tanh", return_sequences=True,
                               input_shape=(self.n_lags, n_features)))
                model.add(LSTM(units // 2, activation="tanh"))

            if dropout > 0:
                model.add(Dropout(dropout))

            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=['mae'])

            return model

        def train_model(self, model: Sequential, X_train: pd.DataFrame, y_train: pd.Series) -> Sequential:
            try:
                # Robust data preparation
                X_values = X_train.values
                y_values = y_train.values

                # Handle any remaining NaN values
                X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
                y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

                # Scale the data
                X_scaled = self.scaler_X.fit_transform(X_values)
                y_scaled = self.scaler_y.fit_transform(y_values.reshape(-1, 1)).flatten()

                # Create sequences
                X_seq, y_seq = create_sequences(X_scaled, y_scaled, self.n_lags)

                if len(X_seq) == 0:
                    print(f"Warning: Not enough data for LSTM sequences (need >{self.n_lags} samples)")
                    return model

                # Training configuration
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=5, restore_best_weights=True, verbose=0
                )

                batch_size = min(32, max(1, len(X_seq) // 4))
                epochs = min(30, self.config.epochs)

                # Train the model
                model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[early_stopping],
                    validation_split=0.2 if len(X_seq) > 10 else 0
                )

                return model

            except Exception as e:
                print(f"Error training Keras LSTM: {e}")
                return model

        def predict(self, model: Sequential, X_test: pd.DataFrame) -> np.ndarray:
            try:
                X_values = X_test.values
                X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
                X_scaled = self.scaler_X.transform(X_values)

                if len(X_scaled) < self.n_lags:
                    # Not enough data for sequences
                    fallback_val = self.scaler_y.inverse_transform([[0]])[0][0]
                    return np.full(len(X_test), max(0, fallback_val))

                X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), self.n_lags)

                if len(X_seq) == 0:
                    return np.full(len(X_test), 0)

                predictions_scaled = model.predict(X_seq, verbose=0)
                predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

                # Handle sequence length mismatch
                if len(predictions) < len(X_test):
                    last_pred = predictions[-1] if len(predictions) > 0 else 0
                    padding = np.full(len(X_test) - len(predictions), last_pred)
                    predictions = np.concatenate([padding, predictions])

                return np.maximum(predictions[:len(X_test)], 0)

            except Exception as e:
                print(f"Error predicting with Keras LSTM: {e}")
                return np.full(len(X_test), 0)


    class KerasGRUModel(KerasLSTMModel):
        """FIXED GRU model using Keras/TensorFlow"""

        def create_model(self, trial: optuna.Trial, n_features: int) -> Sequential:
            max_lags = min(20, n_features, 15)
            self.n_lags = trial.suggest_int('n_lags', 5, max_lags)

            units = trial.suggest_int('units', 32, 128)
            dropout = trial.suggest_float('dropout', 0.0, 0.3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            model = Sequential([
                GRU(units, activation="tanh", input_shape=(self.n_lags, n_features)),
                Dropout(dropout) if dropout > 0 else tf.keras.layers.Lambda(lambda x: x),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=['mae'])
            return model


    class KerasTransformerModel(BaseModel):
        """FIXED Transformer model using Keras/TensorFlow"""

        def __init__(self, config):
            super().__init__(config)
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            self.n_lags = 10

        def create_model(self, trial: optuna.Trial, n_features: int) -> tf.keras.Model:
            max_lags = min(15, n_features, 12)  # Conservative for transformers
            self.n_lags = trial.suggest_int('n_lags', 5, max_lags)

            d_model = trial.suggest_categorical('d_model', [32, 64])
            n_heads = 4  # Fixed to avoid complexity
            ff_dim = trial.suggest_int('ff_dim', 32, 64)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            inputs = Input(shape=(self.n_lags, n_features))

            # Project to d_model dimensions
            x = Dense(d_model)(inputs)

            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model // n_heads
            )(x, x)

            # Add & Norm
            x = LayerNormalization()(attention_output + x)

            # Feed forward
            ff_output = Dense(ff_dim, activation="relu")(x)
            ff_output = Dense(d_model)(ff_output)

            # Add & Norm
            x = LayerNormalization()(ff_output + x)

            # Global average pooling and output
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            outputs = Dense(1)(x)

            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=['mae'])

            return model

        def train_model(self, model: tf.keras.Model, X_train: pd.DataFrame, y_train: pd.Series) -> tf.keras.Model:
            try:
                # Use same robust approach as LSTM
                X_values = X_train.values
                y_values = y_train.values

                X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
                y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

                X_scaled = self.scaler_X.fit_transform(X_values)
                y_scaled = self.scaler_y.fit_transform(y_values.reshape(-1, 1)).flatten()

                X_seq, y_seq = create_sequences(X_scaled, y_scaled, self.n_lags)

                if len(X_seq) == 0:
                    print(f"Warning: Not enough data for Transformer sequences")
                    return model

                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=3, restore_best_weights=True, verbose=0
                )

                batch_size = min(16, max(1, len(X_seq) // 4))
                epochs = min(20, self.config.epochs)

                model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[early_stopping],
                    validation_split=0.2 if len(X_seq) > 10 else 0
                )

                return model

            except Exception as e:
                print(f"Error training Transformer: {e}")
                return model

        def predict(self, model: tf.keras.Model, X_test: pd.DataFrame) -> np.ndarray:
            # Same prediction logic as LSTM
            try:
                X_values = X_test.values
                X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
                X_scaled = self.scaler_X.transform(X_values)

                if len(X_scaled) < self.n_lags:
                    fallback_val = self.scaler_y.inverse_transform([[0]])[0][0]
                    return np.full(len(X_test), max(0, fallback_val))

                X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), self.n_lags)

                if len(X_seq) == 0:
                    return np.full(len(X_test), 0)

                predictions_scaled = model.predict(X_seq, verbose=0)
                predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

                if len(predictions) < len(X_test):
                    last_pred = predictions[-1] if len(predictions) > 0 else 0
                    padding = np.full(len(X_test) - len(predictions), last_pred)
                    predictions = np.concatenate([padding, predictions])

                return np.maximum(predictions[:len(X_test)], 0)

            except Exception as e:
                print(f"Error predicting with Transformer: {e}")
                return np.full(len(X_test), 0)


# ENHANCED PYTORCH MODELS
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.X) - self.sequence_length + 1)

    def __getitem__(self, idx):
        if len(self) == 0:
            return torch.zeros(self.sequence_length, self.X.shape[1]), torch.tensor(0.0)
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length - 1]
        )


class LSTMNetwork(nn.Module):
    """Enhanced LSTM neural network"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output


class TransformerNetwork(nn.Module):
    """PyTorch Transformer neural network"""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super(TransformerNetwork, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling: (batch_size, d_model)
        x = self.dropout(x)
        output = self.fc(x)  # (batch_size, 1)
        return output


class DeepLearningModel(BaseModel):
    """Enhanced PyTorch Deep Learning model implementation"""

    def __init__(self, config, model_type: str = 'lstm'):
        super().__init__(config)
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def create_model(self, trial: optuna.Trial, n_features: int) -> nn.Module:
        if self.model_type == 'lstm':
            return LSTMNetwork(
                input_size=n_features,
                hidden_size=trial.suggest_int('hidden_size', 32, 128),
                num_layers=trial.suggest_int('num_layers', 1, 3),
                dropout=trial.suggest_float('dropout', 0.1, 0.5)
            ).to(self.device)

        elif self.model_type == 'transformer':
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            nhead = trial.suggest_categorical('nhead', [2, 4, 8])
            # Ensure d_model is divisible by nhead
            while d_model % nhead != 0:
                nhead = nhead // 2
                if nhead < 2:
                    nhead = 2
                    break

            return TransformerNetwork(
                input_size=n_features,
                d_model=d_model,
                nhead=nhead,
                num_layers=trial.suggest_int('num_layers', 1, 3),
                dropout=trial.suggest_float('dropout', 0.1, 0.3)
            ).to(self.device)

    def train_model(self, model: nn.Module, X_train: pd.DataFrame, y_train: pd.Series) -> nn.Module:
        sequence_length = min(self.config.sequence_length, len(X_train) // 3)
        sequence_length = max(2, sequence_length)  # Minimum sequence length

        # Robust data preparation
        X_values = X_train.values
        y_values = y_train.values

        X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
        y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)

        train_dataset = TimeSeriesDataset(X_values, y_values, sequence_length)

        if len(train_dataset) == 0:
            print("Warning: Not enough data for PyTorch training")
            return model

        batch_size = min(self.config.batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        model.train()
        epochs = min(self.config.epochs, 30)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)

            # Early stopping check
            if epoch > 5 and avg_loss < 1e-6:
                break

        return model

    def predict(self, model: nn.Module, X_test: pd.DataFrame) -> np.ndarray:
        sequence_length = min(self.config.sequence_length, len(X_test) // 3)
        sequence_length = max(2, sequence_length)

        X_values = X_test.values
        X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

        test_dataset = TimeSeriesDataset(X_values, np.zeros(len(X_values)), sequence_length)

        if len(test_dataset) == 0:
            return np.zeros(len(X_test))

        batch_size = min(self.config.batch_size, len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())

        # Handle length mismatch
        while len(predictions) < len(X_test):
            predictions.append(predictions[-1] if predictions else 0.0)

        return np.maximum(np.array(predictions[:len(X_test)]), 0)


# ADDITIONAL ENSEMBLE MODEL
class EnsembleModel(BaseModel):
    """Ensemble of multiple models"""

    def __init__(self, config, base_models=None):
        super().__init__(config)
        self.base_models = base_models or []
        self.model_weights = None

    def create_model(self, trial: optuna.Trial, n_features: int) -> dict:
        # Create ensemble of different model types
        models = {}

        # Always include these robust models
        models['rf'] = RandomForestModel(self.config)
        models['xgb'] = XGBoostModel(self.config)
        models['ridge'] = RidgeModel(self.config)

        if STATSMODELS_AVAILABLE:
            models['arima'] = ARIMAModel(self.config)

        if TENSORFLOW_AVAILABLE:
            models['lstm'] = KerasLSTMModel(self.config)

        return models

    def train_model(self, models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        trained_models = {}

        for name, model in models.items():
            try:
                print(f"Training ensemble component: {name}")
                # Simple hyperparameter optimization for each model
                mock_trial = self._create_mock_trial()
                created_model = model.create_model(mock_trial, X_train.shape[1])
                trained_model = model.train_model(created_model, X_train, y_train)
                trained_models[name] = {
                    'model': trained_model,
                    'model_instance': model
                }
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                continue

        return trained_models

    def _create_mock_trial(self):
        """Create a mock trial with reasonable default parameters"""

        class MockTrial:
            def suggest_int(self, name, low, high):
                defaults = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_samples_split': 5,
                    'num_layers': 2,
                    'hidden_size': 64,
                    'n_lags': 10
                }
                return defaults.get(name, (low + high) // 2)

            def suggest_float(self, name, low, high, log=False):
                defaults = {
                    'learning_rate': 0.1,
                    'alpha': 1.0,
                    'dropout': 0.2,
                    'lr': 0.001
                }
                return defaults.get(name, (low + high) / 2)

            def suggest_categorical(self, name, choices):
                defaults = {
                    'max_features': 'sqrt',
                    'd_model': 64,
                    'seasonal': False
                }
                return defaults.get(name, choices[0])

        return MockTrial()

    def predict(self, trained_models: dict, X_test: pd.DataFrame) -> np.ndarray:
        predictions = []
        valid_models = []

        for name, model_data in trained_models.items():
            try:
                model_instance = model_data['model_instance']
                model = model_data['model']
                pred = model_instance.predict(model, X_test)

                # Validate predictions
                if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                    predictions.append(pred)
                    valid_models.append(name)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                continue

        if not predictions:
            return np.ones(len(X_test))  # Fallback prediction

        # Simple averaging ensemble
        ensemble_pred = np.mean(predictions, axis=0)
        return np.maximum(ensemble_pred, 0)  # Ensure non-negative


# Model factory function to help with initialization
def get_available_models(config):
    """Return dictionary of all available models"""
    models = {
        'baseline_mean': BaselineModel(config),
        'random_forest': RandomForestModel(config),
        'gradient_boosting': GradientBoostingModel(config),
        'ridge_regression': RidgeModel(config),
        'xgboost': XGBoostModel(config),
        'lightgbm': LightGBMModel(config),
        
        'pytorch_lstm': DeepLearningModel(config, 'lstm'),
        'pytorch_transformer': DeepLearningModel(config, 'transformer'),
        'ensemble': EnsembleModel(config)
            
    }

    # Add time series models if available
    if STATSMODELS_AVAILABLE:
        models['arima'] = ARIMAModel(config)
        models['exponential_smoothing'] = ExponentialSmoothingModel(config)

        if PMDARIMA_AVAILABLE:
            models['auto_arima'] = AutoARIMAModel(config)

    # Add Prophet if available
    if PROPHET_AVAILABLE:
        models['prophet'] = ProphetModel(config)

    # Add Keras models if available
    if TENSORFLOW_AVAILABLE:
        models['keras_lstm'] = KerasLSTMModel(config)
        models['keras_gru'] = KerasGRUModel(config)
        models['keras_transformer'] = KerasTransformerModel(config)

    return models