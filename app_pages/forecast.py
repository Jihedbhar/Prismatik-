import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import logging
from plotly.subplots import make_subplots
import os
import json
from utils.config_Sales_forecasting import ModelConfig
from utils.feature_engineering_Sales_forecasting import FlexibleFeatureEngineer
from utils.models_Sales_forecasting import get_available_models
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split

# Initialize logger
logger = logging.getLogger(__name__)


COLORS = {
    'primary': '#1E3A8A',    # Dark blue
    'secondary': '#3B82F6',  # Lighter blue
    'success': '#10B981',    # Green
    'warning': '#F59E0B',    # Orange
    'error': '#EF4444',      # Red
    'background': '#F8F9FA', # Light gray
    'text': '#1F2937',       # Dark gray
    'accent': '#8B5CF6',     # Purple for trends
}

# Define model availability flags (based on original forecasting code)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Reusing load_dataframe and get_column_mapping
def load_dataframe(table_name):
    """Load DataFrame from session state with error handling"""
    try:
        df = st.session_state.get('dataframes', {}).get(table_name, pd.DataFrame())
        if df.empty:
            logger.warning(f"No DataFrame found for {table_name}")
            return pd.DataFrame()
        logger.info(f"Loaded {table_name} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

def get_column_mapping(table_name, column_name):
    """Retrieve mapped column name, fallback to original if not found"""
    mappings = st.session_state.get('column_mappings', {}).get(table_name, {})
    return mappings.get(column_name, column_name)

def prepare_enriched_data():
    """Prepare a single enriched DataFrame from stored DataFrames"""
    try:
        # Load DataFrames
        data = {
            'Transactions': load_dataframe('Transactions'),
            'Client': load_dataframe('Client'),
            'Produit': load_dataframe('Produit'),
            'Magasin': load_dataframe('Magasin'),
            'Localisation': load_dataframe('Localisation'),
        }

        # Check if Transactions has data
        if data['Transactions'].empty:
            logger.error("No transaction data available")
            return pd.DataFrame(), "❌ No transaction data available"

        enriched_df = data['Transactions'].copy()
        enriched_info = ["Transactions"]

        # Define column lists
        client_cols = ['nom', 'genre', 'âge', 'ville', 'Tier_fidelité', 'premier_achat']
        produit_cols = ['nom_produit', 'catégorie', 'sous_catégorie', 'prix_vente']
        magasin_cols = ['nom_magasin', 'type', 'id_localisation']
        localisation_cols = ['ville', 'gouvernorat', 'pays']

        # Filter only available columns
        client_cols = [col for col in client_cols if get_column_mapping('Client', col) in data['Client'].columns]
        produit_cols = [col for col in produit_cols if get_column_mapping('Produit', col) in data['Produit'].columns]
        magasin_cols = [col for col in magasin_cols if get_column_mapping('Magasin', col) in data['Magasin'].columns]
        localisation_cols = [col for col in localisation_cols if get_column_mapping('Localisation', col) in data['Localisation'].columns]

        # --- Merge with Client ---
        if not data['Client'].empty:
            left_key = get_column_mapping('Transactions', 'id_client')
            right_key = get_column_mapping('Client', 'id_client')
            client_selected = [right_key] + [get_column_mapping('Client', col) for col in client_cols]

            enriched_df = enriched_df.merge(
                data['Client'][client_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Client', col): f'client_{col}' for col in client_cols
            })

            enriched_info.append(f"Client ({len(client_cols)} columns)")

        # --- Merge with Produit ---
        if not data['Produit'].empty:
            left_key = get_column_mapping('Transactions', 'id_produit')
            right_key = get_column_mapping('Produit', 'id_produit')
            produit_selected = [right_key] + [get_column_mapping('Produit', col) for col in produit_cols]

            enriched_df = enriched_df.merge(
                data['Produit'][produit_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Produit', col): f'produit_{col}' for col in produit_cols
            })

            enriched_info.append(f"Produit ({len(produit_cols)} columns)")

        # --- Merge with Magasin ---
        if not data['Magasin'].empty:
            left_key = get_column_mapping('Transactions', 'id_magasin')
            right_key = get_column_mapping('Magasin', 'id_magasin')
            magasin_selected = [right_key] + [get_column_mapping('Magasin', col) for col in magasin_cols]

            enriched_df = enriched_df.merge(
                data['Magasin'][magasin_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Magasin', col): f'magasin_{col}' for col in magasin_cols
            })

            enriched_info.append(f"Magasin ({len(magasin_cols)} columns)")

        # --- Merge with Localisation ---
        if not data['Localisation'].empty and get_column_mapping('Magasin', 'id_localisation') in enriched_df.columns:
            left_key = get_column_mapping('Magasin', 'id_localisation')
            right_key = get_column_mapping('Localisation', 'id_localisation')
            localisation_selected = [right_key] + [get_column_mapping('Localisation', col) for col in localisation_cols]

            enriched_df = enriched_df.merge(
                data['Localisation'][localisation_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Localisation', col): f'localisation_{col}' for col in localisation_cols
            })

            enriched_info.append(f"Localisation ({len(localisation_cols)} columns)")

        # Data validation
        if len(enriched_df) < 10:
            logger.error("Enriched DataFrame has fewer than 10 rows")
            return pd.DataFrame(), "❌ Insufficient data: Fewer than 10 rows"

        # Final summary
        summary = f"📊 Enriched Data: {' + '.join(enriched_info)}"
        logger.info(f"Enriched data created: {len(enriched_df)} rows, {len(enriched_df.columns)} columns")

        return enriched_df, summary

    except Exception as e:
        logger.error(f"Error preparing enriched data: {e}")
        return pd.DataFrame(), f"❌ Error: {str(e)}"

class MockTrial:
    """Mock trial object for creating models with best parameters"""
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high):
        return self.params.get(name, (low + high) // 2)

    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, (low + high) / 2)

    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])

class EnhancedCSVSalesForecaster:
    """Enhanced sales forecasting pipeline using enriched DataFrame"""
    def __init__(self, model_config: ModelConfig, enriched_df: pd.DataFrame):
        self.model_config = model_config
        self.enriched_df = enriched_df
        self.feature_engineer = FlexibleFeatureEngineer(model_config)
        self.models = {}
        self.results = {}
        self.original_data = None

    def initialize_models(self, model_selection='all'):
        """Initialize models based on selection criteria"""
        all_models = get_available_models(self.model_config)

        if model_selection == 'all':
            self.models = all_models
        elif model_selection == 'time_series':
            time_series_models = {
                'baseline_mean': all_models.get('baseline_mean'),
                'arima': all_models.get('arima') if STATSMODELS_AVAILABLE else None,
                'auto_arima': all_models.get('auto_arima') if PMDARIMA_AVAILABLE else None,
                'exponential_smoothing': all_models.get('exponential_smoothing') if STATSMODELS_AVAILABLE else None,
                'prophet': all_models.get('prophet') if PROPHET_AVAILABLE else None,
                'keras_lstm': all_models.get('keras_lstm') if TENSORFLOW_AVAILABLE else None,
                'keras_gru': all_models.get('keras_gru') if TENSORFLOW_AVAILABLE else None,
                'pytorch_lstm': all_models.get('pytorch_lstm')
            }
            self.models = {k: v for k, v in time_series_models.items() if v is not None}
        elif model_selection == 'transformers':
            transformer_models = {
                'baseline_mean': all_models.get('baseline_mean'),
                'keras_transformer': all_models.get('keras_transformer') if TENSORFLOW_AVAILABLE else None,
                'pytorch_transformer': all_models.get('pytorch_transformer'),
                'random_forest': all_models.get('random_forest'),
                'xgboost': all_models.get('xgboost')
            }
            self.models = {k: v for k, v in transformer_models.items() if v is not None}
        elif model_selection == 'fast':
            fast_models = {
                'baseline_mean': all_models.get('baseline_mean'),
                'random_forest': all_models.get('random_forest'),
                'ridge_regression': all_models.get('ridge_regression'),
                'arima': all_models.get('arima') if STATSMODELS_AVAILABLE else None,
                'keras_lstm': all_models.get('keras_lstm') if TENSORFLOW_AVAILABLE else None
            }
            self.models = {k: v for k, v in fast_models.items() if v is not None}
        else:
            self.models = {k: all_models[k] for k in model_selection if k in all_models}

        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")

    def load_data(self) -> pd.DataFrame:
        """Return the enriched DataFrame"""
        if self.enriched_df.empty:
            logger.error("Enriched DataFrame is empty")
            return pd.DataFrame()
        return self.enriched_df

    def train_and_evaluate_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate all models with enhanced error handling"""
        logger.info(f"\n📈 Dataset Analysis:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df[self.model_config.time_column].min()} to {df[self.model_config.time_column].max()}")
        logger.info(f"  Target variable stats: {df[self.model_config.target_column].describe()}")

        # Store original data
        self.original_data = df.copy()

        try:
            # Validate date and target columns
            if self.model_config.time_column not in df.columns:
                raise ValueError(f"Time column '{self.model_config.time_column}' not found in data")
            if self.model_config.target_column not in df.columns:
                raise ValueError(f"Target column '{self.model_config.target_column}' not found in data")
            
            # Convert date column to datetime
            try:
                df[self.model_config.time_column] = pd.to_datetime(df[self.model_config.time_column])
            except Exception as e:
                raise ValueError(f"Failed to convert '{self.model_config.time_column}' to datetime: {e}")

            # Ensure target column is numeric
            try:
                df[self.model_config.target_column] = pd.to_numeric(df[self.model_config.target_column], errors='coerce')
                if df[self.model_config.target_column].isna().any():
                    raise ValueError(f"Target column '{self.model_config.target_column}' contains non-numeric or missing values")
            except Exception as e:
                raise ValueError(f"Failed to convert '{self.model_config.target_column}' to numeric: {e}")

            # Prepare data with enhanced feature engineering
            X, y, feature_cols = self.feature_engineer.prepare_data(df)
            logger.info(f"\n🔧 Feature Engineering Results:")
            logger.info(f"  Generated features: {len(feature_cols)}")
            logger.info(f"  Final dataset shape: {X.shape}")
            logger.info(f"  Feature examples: {feature_cols[:5]}...")

            if len(X) < 10:
                raise ValueError("Insufficient data for training (need at least 10 samples)")

            # Enhanced time-based splitting for time series
            if self.model_config.time_column in df.columns:
                # Sort by time to ensure no data leakage
                df_sorted = df.sort_values(self.model_config.time_column)
                sort_idx = df_sorted.index
                X = X.reindex(sort_idx)
                y = y.reindex(sort_idx)

                # Chronological split
                split_idx = int(len(X) * (1 - self.model_config.test_size))
                X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                # Validation split (also chronological)
                val_split_idx = int(len(X_train_full) * 0.8)
                X_train, X_val = X_train_full.iloc[:val_split_idx], X_train_full.iloc[val_split_idx:]
                y_train, y_val = y_train_full.iloc[:val_split_idx], y_train_full.iloc[val_split_idx:]

                logger.info("✅ Using TIME-BASED splitting (no data leakage)")
            else:
                # Random split if no time column
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.model_config.test_size,
                    random_state=self.model_config.random_state, shuffle=False
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2,
                    random_state=self.model_config.random_state, shuffle=False
                )

            logger.info(f"📊 Data Split: Train({X_train.shape[0]}) | Val({X_val.shape[0]}) | Test({X_test.shape[0]})")

            # Store test data for later predictions
            self.X_test = X_test
            self.y_test = y_test

            logger.info(f"\n🚀 Starting enhanced model training...")

            # Track model training time and success rate
            successful_models = 0
            total_models = len(self.models)

            for model_name, model in self.models.items():
                logger.info(f"\n  🔄 Training {model_name}...")
                start_time = datetime.now()

                try:
                    # Enhanced hyperparameter optimization with error handling
                    try:
                        optimization_result = model.optimize_hyperparameters(
                            X_train, y_train, X_val, y_val
                        )
                        logger.info(f"    ⚡ Optimization: {optimization_result['n_trials']} trials, "
                                    f"best score: {optimization_result['best_score']:.4f}")
                    except Exception as opt_error:
                        logger.warning(f"    ⚠️ Optimization failed: {opt_error}, using defaults")
                        optimization_result = {
                            'best_params': {},
                            'best_score': float('inf'),
                            'n_trials': 0
                        }

                    # Train final model with best parameters
                    mock_trial = MockTrial(optimization_result['best_params'])
                    final_model = model.create_model(mock_trial, X_train.shape[1])

                    # Use full training data (train + validation)
                    X_final_train = pd.concat([X_train, X_val])
                    y_final_train = pd.concat([y_train, y_val])

                    trained_model = model.train_model(final_model, X_final_train, y_final_train)

                    # Make predictions with error handling
                    try:
                        y_pred = model.predict(trained_model, X_test)

                        # Validate predictions
                        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                            logger.warning(f"    ⚠️ Invalid predictions detected, cleaning...")
                            y_pred = np.nan_to_num(y_pred, nan=y_train.mean(), posinf=y_train.max(), neginf=0)

                        # Ensure non-negative predictions for sales
                        y_pred = np.maximum(y_pred, 0)

                    except Exception as pred_error:
                        logger.error(f"    ❌ Prediction failed: {pred_error}")
                        y_pred = np.full(len(X_test), y_train.mean())

                    # Evaluate model performance
                    evaluation = model.evaluate_model(y_test.values, y_pred)

                    # Calculate training time
                    training_time = (datetime.now() - start_time).total_seconds()

                    # Store comprehensive results
                    self.results[model_name] = {
                        'model': trained_model,
                        'model_instance': model,
                        'best_params': optimization_result['best_params'],
                        'best_score': optimization_result['best_score'],
                        'evaluation': evaluation,
                        'feature_importance': getattr(model, 'feature_importance', None),
                        'predictions': y_pred.tolist(),
                        'training_time': training_time,
                        'n_trials': optimization_result['n_trials']
                    }

                    logger.info(f"    ✅ {model_name} - RMSE: {evaluation['rmse']:.4f}, "
                                f"MAE: {evaluation['mae']:.4f}, R²: {evaluation['r2']:.4f}, "
                                f"MAPE: {evaluation['mape']:.2f}%, Time: {training_time:.1f}s")

                    successful_models += 1

                except Exception as e:
                    logger.error(f"    ❌ Error training {model_name}: {str(e)}")
                    # Store error information
                    self.results[model_name] = {
                        'error': str(e),
                        'evaluation': {
                            'rmse': float('inf'), 'mae': float('inf'),
                            'r2': -float('inf'), 'mape': float('inf')
                        },
                        'training_time': (datetime.now() - start_time).total_seconds()
                    }
                    continue

            logger.info(f"\n📊 Training Summary: {successful_models}/{total_models} models trained successfully")
            return self.results

        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {e}")
            raise e

    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """Get the best performing model based on RMSE"""
        if not self.results:
            logger.warning("No results available")
            return None, None

        valid_results = {k: v for k, v in self.results.items()
                         if 'evaluation' in v and v['evaluation']['rmse'] != float('inf')}

        if not valid_results:
            logger.warning("No valid models found")
            return None, None

        best_model_name = min(valid_results.keys(),
                              key=lambda x: valid_results[x]['evaluation']['rmse'])
        return best_model_name, valid_results[best_model_name]

    def make_future_predictions(self, periods: int = 6) -> pd.DataFrame:
        """Make future predictions using the best model with daily dates"""
        best_model_name, best_result = self.get_best_model()

        if not best_model_name:
            logger.error("No valid model found for predictions")
            return pd.DataFrame()

        logger.info(f"Making {periods} monthly periods ({periods * 30} daily predictions) using {best_model_name}...")

        try:
            # Get unique store/product combinations from original data
            store_product_combinations = self.original_data[
                [self.model_config.store_column, self.model_config.product_column]
            ].drop_duplicates()

            logger.info(f"Found {len(store_product_combinations)} store-product combinations")

            # Get the last date from original data and set to next day
            if self.model_config.time_column in self.original_data.columns:
                last_date = pd.to_datetime(self.original_data[self.model_config.time_column]).max()
                # Set to next day, strip time components
                last_date = (last_date + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                # Calculate days for the specified number of months (approximate)
                days = periods * 30  # Approximate 30 days per month
                future_dates = pd.date_range(start=last_date, periods=days, freq='D')
            else:
                # Fallback to a default start date
                last_date = pd.to_datetime('2024-01-01').replace(hour=0, minute=0, second=0, microsecond=0)
                days = periods * 30
                future_dates = pd.date_range(start=last_date, periods=days, freq='D')

            logger.info(f"Generating predictions for {len(future_dates)} days from {future_dates[0]} to {future_dates[-1]}")

            all_predictions = []

            # For each store-product combination
            for _, combo in store_product_combinations.iterrows():
                store_id = combo[self.model_config.store_column]
                product_id = combo[self.model_config.product_column]

                # Get historical data for this store-product combination
                historical_data = self.original_data[
                    (self.original_data[self.model_config.store_column] == store_id) &
                    (self.original_data[self.model_config.product_column] == product_id)
                ].copy()

                if len(historical_data) == 0:
                    continue

                # Get the last known values for this combination
                last_row = historical_data.iloc[-1:].copy()

                # Create future data for this store-product combination
                future_data = []
                for date in future_dates:
                    future_row = last_row.copy()
                    future_row[self.model_config.time_column] = date
                    future_row[self.model_config.store_column] = store_id
                    future_row[self.model_config.product_column] = product_id
                    future_data.append(future_row)

                future_df = pd.concat(future_data, ignore_index=True)

                # Combine historical and future data for feature engineering
                combined_data = pd.concat([historical_data, future_df], ignore_index=True)

                try:
                    # Prepare features
                    X_combined, _, _ = self.feature_engineer.prepare_data(combined_data)

                    # Get only the future part
                    X_future = X_combined.iloc[-len(future_dates):]

                    # Make predictions using the best model
                    if len(X_future) > 0:
                        model_instance = best_result['model_instance']
                        trained_model = best_result['model']

                        future_predictions = model_instance.predict(trained_model, X_future)

                        # Create results for this store-product combination
                        for i, (date, prediction) in enumerate(zip(future_dates, future_predictions)):
                            all_predictions.append({
                                'date': date.strftime('%d/%m/%Y'),  # Format as DD/MM/YYYY
                                'store_id': store_id,
                                'product_id': product_id,
                                'predicted_sales': max(0, prediction),
                                'model_used': best_model_name,
                                'confidence': 'high' if best_result['evaluation']['r2'] > 0.5 else 'medium' if
                                            best_result['evaluation']['r2'] > 0.2 else 'low',
                                'prediction_timestamp': datetime.now()
                            })

                except Exception as e:
                    logger.error(f"Error predicting for store {store_id}, product {product_id}: {e}")
                    # Add fallback predictions
                    historical_avg = historical_data[self.model_config.target_column].mean()
                    for date in future_dates:
                        all_predictions.append({
                            'date': date.strftime('%d/%m/%Y'),  # Format as DD/MM/YYYY
                            'store_id': store_id,
                            'product_id': product_id,
                            'predicted_sales': max(0, historical_avg),
                            'model_used': f'{best_model_name}_fallback',
                            'confidence': 'low',
                            'prediction_timestamp': datetime.now()
                        })

            if all_predictions:
                predictions_df = pd.DataFrame(all_predictions)
                logger.info(f"Generated {len(predictions_df)} predictions")
                logger.info(f"Average predicted sales: {predictions_df['predicted_sales'].mean():.2f}")
                logger.info(f"Stores: {predictions_df['store_id'].nunique()}")
                logger.info(f"Products: {predictions_df['product_id'].nunique()}")
                return predictions_df
            else:
                logger.warning("No predictions generated")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error making future predictions: {e}")
            return pd.DataFrame()

    def save_results(self, save_dir: str = 'results') -> str:
        """Save all results and models with enhanced organization"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save test predictions with actual values
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            test_indices = self.y_test.index
            original_test_data = self.original_data.iloc[test_indices].copy()

            test_predictions = pd.DataFrame({
                'date': original_test_data[self.model_config.time_column],
                'store_id': original_test_data[self.model_config.store_column],
                'product_id': original_test_data[self.model_config.product_column],
                'actual_sales': self.y_test.values
            })

            # Add predictions from all successful models
            for model_name, result in self.results.items():
                if 'predictions' in result and len(result['predictions']) == len(self.y_test):
                    test_predictions[f'{model_name}_predicted'] = result['predictions']

            test_predictions_path = os.path.join(save_dir, 'test_predictions_with_actuals.csv')
            test_predictions.to_csv(test_predictions_path, index=False)
            logger.info(f"Saved test predictions: {test_predictions_path}")

        # 2. Save future predictions
        future_predictions = self.make_future_predictions(30)
        if not future_predictions.empty:
            future_path = os.path.join(save_dir, 'future_predictions.csv')
            future_predictions.to_csv(future_path, index=False)
            logger.info(f"Saved future predictions: {future_path}")

        # 3. Save comprehensive model performance comparison
        performance_data = []
        for model_name, result in self.results.items():
            if 'evaluation' in result:
                eval_data = result['evaluation'].copy()
                eval_data.update({
                    'model_name': model_name,
                    'training_time': result.get('training_time', 0),
                    'n_trials': result.get('n_trials', 0),
                    'has_error': 'error' in result,
                    'error_message': result.get('error', '')
                })
                performance_data.append(eval_data)

        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('rmse')
            performance_path = os.path.join(save_dir, 'model_performance_detailed.csv')
            performance_df.to_csv(performance_path, index=False)
            logger.info(f"Saved detailed model performance: {performance_path}")

        # 4. Save feature importance from best model
        best_model_name, best_result = self.get_best_model()
        if best_result and best_result.get('feature_importance'):
            feature_importance = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in best_result['feature_importance'].items()
            ]).sort_values('importance', ascending=False)

            importance_path = os.path.join(save_dir, f'{best_model_name}_feature_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Saved feature importance: {importance_path}")

        # 5. Save model configuration and metadata
        metadata = {
            'config': {
                'model_config': vars(self.model_config),
                # Removed csv_config reference
                'columns': {
                    'time_column': self.model_config.time_column,
                    'store_column': self.model_config.store_column,
                    'product_column': self.model_config.product_column,
                    'target_column': self.model_config.target_column
                }
            },
            'dataset_info': {
                'total_records': len(self.original_data) if self.original_data is not None else 0,
                'date_range': {
                    'start': str(self.original_data[self.model_config.time_column].min()) if self.original_data is not None else None,
                    'end': str(self.original_data[self.model_config.time_column].max()) if self.original_data is not None else None
                },
                'unique_stores': self.original_data[self.model_config.store_column].nunique() if self.original_data is not None else 0,
                'unique_products': self.original_data[self.model_config.product_column].nunique() if self.original_data is not None else 0
            },
            'best_model': {
                'name': best_model_name,
                'performance': best_result['evaluation'] if best_result else None
            },
            'generation_timestamp': datetime.now().isoformat()
        }

        metadata_path = os.path.join(save_dir, 'forecasting_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {metadata_path}")

        logger.info(f"All results saved to: {save_dir}")
        return save_dir

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report with model comparisons"""
        if not self.results:
            return "No results available"

        report = [
            "Enhanced CSV Sales Forecasting Report",
            "=" * 60,
            ""
        ]

        # Dataset summary
        if self.original_data is not None:
            report.extend([
                "📊 Dataset Summary:",
                f"  Total records: {len(self.original_data):,}",
                f"  Date range: {self.original_data[self.model_config.time_column].min()} to {self.original_data[self.model_config.time_column].max()}",
                f"  Unique stores: {self.original_data[self.model_config.store_column].nunique()}",
                f"  Unique products: {self.original_data[self.model_config.product_column].nunique()}",
                f"  Average daily sales: {self.original_data[self.model_config.target_column].mean():.2f}",
                f"  Sales std deviation: {self.original_data[self.model_config.target_column].std():.2f}",
                f"  Available columns: {len(self.original_data.columns)}",
                ""
            ])

        # Feature engineering summary
        if hasattr(self.feature_engineer, 'feature_columns'):
            report.extend([
                "🔧 Feature Engineering:",
                f"  Generated features: {len(self.feature_engineer.feature_columns)}",
                f"  Feature categories: Time-based, Product-based, Store-based, Customer-based, Interaction features",
                f"  Lag features: 1-day, 7-day lags with rolling statistics",
                ""
            ])

        # Model performance table
        report.extend([
            "🏆 Model Performance Comparison:",
            "-" * 100,
            f"{'Model':<25} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8} | {'MAPE':<8} | {'Time(s)':<8} | {'Status':<10}",
            "-" * 100
        ])

        # Sort models by performance
        valid_results = [(k, v) for k, v in self.results.items()
                         if 'evaluation' in v and v['evaluation']['rmse'] != float('inf')]
        valid_results.sort(key=lambda x: x[1]['evaluation']['rmse'])

        for model_name, result in valid_results:
            eval_metrics = result['evaluation']
            training_time = result.get('training_time', 0)
            status = "✅ Success" if 'error' not in result else "❌ Failed"

            report.append(
                f"{model_name:<25} | {eval_metrics['rmse']:<8.4f} | "
                f"{eval_metrics['mae']:<8.4f} | {eval_metrics['r2']:<8.4f} | "
                f"{eval_metrics['mape']:<8.2f}% | {training_time:<8.1f} | {status:<10}"
            )

        # Add failed models
        failed_results = [(k, v) for k, v in self.results.items() if 'error' in v]
        for model_name, result in failed_results:
            training_time = result.get('training_time', 0)
            report.append(
                f"{model_name:<25} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | "
                f"{'N/A':<8} | {training_time:<8.1f} | ❌ Failed"
            )

        report.append("")

        # Best model details
        best_name, best_result = self.get_best_model()
        if best_name:
            report.extend([
                f"🥇 Best Model: {best_name}",
                f"📈 Performance Metrics:",
                f"   RMSE: {best_result['evaluation']['rmse']:.4f}",
                f"   MAE: {best_result['evaluation']['mae']:.4f}",
                f"   R²: {best_result['evaluation']['r2']:.4f}",
                f"   MAPE: {best_result['evaluation']['mape']:.2f}%",
                f"   Training time: {best_result.get('training_time', 0):.1f}s",
                ""
            ])

            # Model-specific insights
            if 'time_series' in best_name.lower() or 'arima' in best_name.lower() or 'prophet' in best_name.lower():
                report.extend([
                    "📊 Time Series Model Insights:",
                    "   ✓ Captures temporal patterns and seasonality",
                    "   ✓ Good for forecasting future trends",
                    "   ✓ Handles irregular time series patterns",
                    ""
                ])
            elif 'transformer' in best_name.lower():
                report.extend([
                    "🤖 Transformer Model Insights:",
                    "   ✓ Captures complex sequential dependencies",
                    "   ✓ Attention mechanism for pattern recognition",
                    "   ✓ State-of-the-art for sequence modeling",
                    ""
                ])
            elif 'lstm' in best_name.lower() or 'gru' in best_name.lower():
                report.extend([
                    "🧠 Neural Network Insights:",
                    "   ✓ Captures long-term dependencies",
                    "   ✓ Learns complex non-linear patterns",
                    "   ✓ Good for sequential data modeling",
                    ""
                ])

            # Feature importance
            if best_result.get('feature_importance'):
                report.extend([
                    "🎯 Top 10 Important Features:",
                ])
                sorted_features = sorted(best_result['feature_importance'].items(),
                                         key=lambda x: abs(x[1]), reverse=True)[:10]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    report.append(f"   {i:2d}. {feature}: {importance:.4f}")
                report.append("")

        # Model recommendations
        report.extend([
            "💡 Recommendations:",
            "   • Time series models work best for trend forecasting",
            "   • Transformer models excel with large datasets and complex patterns",
            "   • Ensemble methods can improve robustness",
            "   • Regular model retraining recommended for production use",
            "   • Monitor prediction confidence levels",
            "",
            "📊 Success Rate:",
            f"   Successfully trained: {len(valid_results)} out of {len(self.results)} models",
            f"   Success rate: {len(valid_results) / len(self.results) * 100:.1f}%"
        ])

        return "\n".join(report)

    def run_model_comparison(self, model_selection='all'):
        """Run a comprehensive model comparison"""
        logger.info(f"Running Enhanced Sales Forecasting Pipeline")
        logger.info(f"Model selection: {model_selection}")

        # Initialize models
        self.initialize_models(model_selection)

        # Load data
        df = self.load_data()
        if df.empty:
            logger.error("No data loaded. Exiting...")
            return None, None

        # Train and evaluate models
        results = self.train_and_evaluate_models(df)

        # Generate and print report
        report = self.generate_comprehensive_report()
        logger.info("\n" + report)

        # Save results
        best_model_name, best_result = self.get_best_model()
        if best_model_name:
            logger.info(f"Best performing model: {best_model_name}")
            results_dir = self.save_results()
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Check {results_dir}/ for all outputs")
        else:
            logger.error("No valid models were trained successfully.")

        return self, results

@st.cache_data
def cached_prepare_enriched_data():
    """Cache the enriched DataFrame to improve performance"""
    return prepare_enriched_data()

def forecasting_page():
    st.set_page_config(page_title="Prismatik Sales Forecasting", layout="wide")
    st.title("📈 Sales Forecasting Dashboard", anchor=False)
    st.markdown("Forecast sales using enriched data with advanced machine learning models.")

    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
        st.session_state.results = None
        st.session_state.model_selection = 'fast'

    with st.sidebar:
        st.header("⚙️ Configuration")
        enriched_df, summary = cached_prepare_enriched_data()
        if enriched_df.empty:
            st.error(summary)
            return

        st.write(summary)
        available_columns = enriched_df.columns.tolist()
        
        date_col = st.selectbox(
            "Date Column",
            options=available_columns,
            index=available_columns.index(get_column_mapping('Transactions', 'date')) if get_column_mapping('Transactions', 'date') in available_columns else 0
        )
        store_col = st.selectbox(
            "Store Column",
            options=available_columns,
            index=available_columns.index(get_column_mapping('Transactions', 'id_magasin')) if get_column_mapping('Transactions', 'id_magasin') in available_columns else 0
        )
        product_col = st.selectbox(
            "Product Column",
            options=available_columns,
            index=available_columns.index(get_column_mapping('Transactions', 'id_produit')) if get_column_mapping('Transactions', 'id_produit') in available_columns else 0
        )
        target_col = st.selectbox(
            "Sales Column",
            options=available_columns,
            index=available_columns.index(get_column_mapping('Transactions', 'quantité_vendue')) if get_column_mapping('Transactions', 'quantité_vendue') in available_columns else 0
        )

        available_models = [
            'all', 'time_series', 'transformers', 'fast',
            'baseline_mean', 'random_forest', 'gradient_boosting',
            'ridge_regression', 'xgboost', 'lightgbm',
            'pytorch_lstm', 'pytorch_transformer', 'ensemble'
        ]
        if STATSMODELS_AVAILABLE:
            available_models.extend(['arima', 'exponential_smoothing'])
            if PMDARIMA_AVAILABLE:
                available_models.append('auto_arima')
        if PROPHET_AVAILABLE:
            available_models.append('prophet')
        if TENSORFLOW_AVAILABLE:
            available_models.extend(['keras_lstm', 'keras_gru', 'keras_transformer'])

        model_selection = st.multiselect(
            "Select Models",
            options=available_models,
            default=['ridge_regression'],  # Default to ridge_regression for testing
            help="Choose specific models or model groups to train."
        )
        if 'all' in model_selection:
            model_selection = ['all']
            st.warning("Selecting 'all' models may take significant time.")

        forecast_periods = st.slider(
            "Forecast Periods (Months)",
            min_value=1,
            max_value=24,
            value=6,
            step=1
        )
        optuna_trials = st.slider(
            "Optimization Trials",
            min_value=10,
            max_value=100,
            value=20,
            step=5
        )
        optuna_timeout = st.slider(
            "Optimization Timeout (Seconds)",
            min_value=300,
            max_value=1800,
            value=600,
            step=300
        )
        run_forecast = st.button("🚀 Run Forecasting", type="primary")

        # Add cache-clearing button
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.session_state.forecaster = None
            st.session_state.results = None
            st.success("Cache cleared. Please rerun forecasting.")

    st.header("📈 Model Training")
    if run_forecast and not enriched_df.empty:
        with st.spinner("Training models..."):
            try:
                model_config = ModelConfig(
                    optuna_trials=optuna_trials,
                    optuna_timeout=optuna_timeout,
                    sequence_length=12,
                    epochs=50,
                    batch_size=32,
                    test_size=0.2,
                    random_state=42,
                    time_column=date_col,
                    store_column=store_col,
                    product_column=product_col,
                    target_column=target_col
                )
                st.session_state.forecaster = EnhancedCSVSalesForecaster(model_config, enriched_df)
                st.session_state.model_selection = model_selection if model_selection else ['ridge_regression']

                from statsmodels.tsa.stattools import adfuller
                result = adfuller(enriched_df[target_col])
                if result[1] > 0.05:
                    st.warning("Data is non-stationary (p > 0.05). Time series models may perform poorly.")
                
                progress_bar = st.progress(0)
                forecaster, results = st.session_state.forecaster.run_model_comparison(st.session_state.model_selection)
                st.session_state.results = results
                logger.info(f"Results after training: {list(results.keys())}")
                st.write(f"Debug: Trained models: {list(results.keys())}")  # Debug output
                progress_bar.progress(100)
                st.success("Training completed successfully!")
            except Exception as e:
                logger.error(f"Training failed: {e}")
                st.error(f"Error during training: {e}")
                progress_bar.progress(100)

    if st.session_state.results:
        st.subheader("Model Performance")
        logger.info(f"Displaying results for models: {list(st.session_state.results.keys())}")
        performance_data = []
        for model_name, result in st.session_state.results.items():
            logger.info(f"Processing result for {model_name}: {result}")
            if 'evaluation' in result:
                eval_data = result['evaluation'].copy()
                eval_data.update({
                    'Model': model_name,
                    'Training Time (s)': result.get('training_time', 0),
                    'Status': 'Success' if 'error' not in result else f"Failed: {result.get('error', 'Unknown')}"
                })
            else:
                eval_data = {
                    'Model': model_name,
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'r2': -float('inf'),
                    'mape': float('inf'),
                    'Training Time (s)': result.get('training_time', 0),
                    'Status': f"Failed: {result.get('error', 'No evaluation data')}"
                }
            performance_data.append(eval_data)

        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('rmse')
            st.write("**Performance Table**")
            st.dataframe(
                performance_df[['Model', 'rmse', 'mae', 'r2', 'mape', 'Training Time (s)', 'Status']],
                use_container_width=True
            )
        else:
            st.error("No performance data available. Check logs for details.")
            logger.error("Performance data is empty")

        # Feature Importance for Best Model
        best_model_name, best_result = st.session_state.forecaster.get_best_model()
        if best_result and best_result.get('feature_importance'):
            st.subheader(f"Feature Importance ({best_model_name})")
            feature_importance = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in best_result['feature_importance'].items()
            ]).sort_values('Importance', ascending=False)
            try:
                fig = px.bar(
                    feature_importance.head(10),
                    x='Feature',
                    y='Importance',
                    title="Top 10 Feature Importance",
                    color_discrete_sequence=[COLORS['accent']]
                )
                fig.update_layout(
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font_color=COLORS['text']
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting feature importance: {e}")
                logger.error(f"Error plotting feature importance: {e}")
    else:
        st.warning("No results to display. Please run forecasting.")

    st.header("🔮 Future Predictions")
    if st.session_state.results and st.button("Generate Predictions", type="secondary"):
        with st.spinner(f"Generating {forecast_periods}-month predictions..."):
            try:
                predictions_df = st.session_state.forecaster.make_future_predictions(periods=forecast_periods)
                if not predictions_df.empty:
                    st.write(f"Debug: Predictions generated for {len(predictions_df)} rows")  # Debug output
                    st.dataframe(predictions_df, use_container_width=True)
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="future_sales_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No predictions generated.")
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                st.error(f"Error generating predictions: {e}")

    

if __name__ == "__main__":
    forecasting_page()