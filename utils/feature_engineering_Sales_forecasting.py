# feature_engineering_Sales_forecasting_flexible.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple


class FlexibleFeatureEngineer:
    """Handle feature engineering with flexible column requirements"""

    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.available_columns = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from available columns only"""
        df = df.copy()
        self.available_columns = df.columns.tolist()

        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found")

        print(f"Available columns for feature engineering: {len(self.available_columns)}")

        # Create features step by step, only if columns exist
        df = self._create_time_features(df)
        df = self._create_product_features(df)
        df = self._create_store_features(df)
        df = self._create_location_features(df)
        df = self._create_customer_features(df)
        df = self._create_temporal_features(df)
        df = self._create_interaction_features(df)

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        if self.config.time_column in df.columns:
            print("  Creating time features...")
            df[self.config.time_column] = pd.to_datetime(df[self.config.time_column])
            df['year'] = df[self.config.time_column].dt.year
            df['month'] = df[self.config.time_column].dt.month
            df['day'] = df[self.config.time_column].dt.day
            df['dayofweek'] = df[self.config.time_column].dt.dayofweek
            df['quarter'] = df[self.config.time_column].dt.quarter
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            df['is_month_start'] = df[self.config.time_column].dt.is_month_start.astype(int)
            df['is_month_end'] = df[self.config.time_column].dt.is_month_end.astype(int)
            df['hour'] = df[self.config.time_column].dt.hour
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        return df

    def _create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product-related features from available columns"""
        print("  Creating product features...")

        # Handle categorical product columns
        categorical_cols = ['category', 'subcategory']
        for col in categorical_cols:
            if col in df.columns:
                df = self._encode_categorical(df, col)
                print(f"    Encoded {col}")

        # Create price-related features if available
        if 'price' in df.columns:
            df['price'] = df['price'].fillna(df['price'].median())
            df['price_category'] = pd.cut(
                df['price'],
                bins=[0, 5, 10, 20, float('inf')],
                labels=['Low', 'Medium', 'High', 'Premium']
            ).astype(str)
            df = self._encode_categorical(df, 'price_category')
            print(f"    Created price categories")

        # Create profit features only if both price and base_cost exist
        if 'base_cost' in df.columns and 'price' in df.columns:
            df['base_cost'] = df['base_cost'].fillna(df['base_cost'].median())
            df['profit_margin'] = (df['price'] - df['base_cost']) / (df['price'] + 1e-8)
            df['markup_ratio'] = df['price'] / (df['base_cost'] + 1e-8)
            df['profit_amount'] = df['price'] - df['base_cost']
            print(f"    Created profit features")

        return df

    def _create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-related features from available columns"""
        print("  Creating store features...")

        # Store type encoding
        if 'store_type' in df.columns:
            df = self._encode_categorical(df, 'store_type')
            print(f"    Encoded store_type")

        # Store amenities (only if available)
        amenity_cols = ['has_wifi', 'has_AC']
        available_amenities = [col for col in amenity_cols if col in df.columns]
        if available_amenities:
            for col in available_amenities:
                df[col] = df[col].fillna(0).astype(int)
            print(f"    Processed amenities: {available_amenities}")

        # Store area features (only if available)
        if 'store_area' in df.columns:
            df['store_area'] = df['store_area'].fillna(df['store_area'].median())
            df['store_size_category'] = pd.cut(
                df['store_area'],
                bins=[0, 200, 500, 1000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'XLarge']
            ).astype(str)
            df = self._encode_categorical(df, 'store_size_category')
            print(f"    Created store size categories")

        # Create store ID features (always available)
        if self.config.store_column in df.columns:
            # Store popularity (transaction count)
            store_counts = df.groupby(self.config.store_column).size()
            df['store_popularity'] = df[self.config.store_column].map(store_counts)
            print(f"    Created store popularity feature")

        return df

    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-related features from available columns"""
        print("  Creating location features...")

        location_cols = ['city', 'state', 'country']
        available_locations = [col for col in location_cols if col in df.columns]

        for col in available_locations:
            df = self._encode_categorical(df, col)
            print(f"    Encoded {col}")

        return df

    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-related features from available columns"""
        print("  Creating customer features...")

        # Age features (if available)
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median())
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['Young', 'Adult', 'MiddleAge', 'Senior', 'Elder']
            ).astype(str)
            df = self._encode_categorical(df, 'age_group')
            print(f"    Created age groups")

        # Gender encoding (if available)
        if 'gender' in df.columns:
            df = self._encode_categorical(df, 'gender')
            print(f"    Encoded gender")

        # Loyalty features (if available)
        if 'is_loyalty_program_member' in df.columns:
            df['is_loyalty_program_member'] = df['is_loyalty_program_member'].fillna(0)
            df['is_loyalty_program_member'] = pd.to_numeric(df['is_loyalty_program_member'], errors='coerce').fillna(
                0).astype(int)
            print(f"    Processed loyalty membership")

        # Customer ID features (always available)
        if 'customer_id' in df.columns:
            # Customer frequency
            customer_counts = df.groupby('customer_id').size()
            df['customer_frequency'] = df['customer_id'].map(customer_counts)
            print(f"    Created customer frequency feature")

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag and rolling window features WITHOUT data leakage"""
        print("  Creating temporal features...")

        if self.config.time_column in df.columns:
            df = df.sort_values([self.config.store_column, self.config.product_column, self.config.time_column])

            # Lag features (proper - no leakage)
            for lag in [1, 7]:  # Reduced lags to be conservative
                lag_col = f'lag_{lag}'
                df[lag_col] = df.groupby([self.config.store_column, self.config.product_column])[
                    self.config.target_column].shift(lag)
                print(f"    Created {lag_col}")

            # Rolling features (proper - only past data)
            for window in [7]:  # Only 7-day window to keep it simple
                shifted_values = df.groupby([self.config.store_column, self.config.product_column])[
                    self.config.target_column].shift(1)

                df[f'rolling_mean_{window}'] = shifted_values.groupby(
                    [df[self.config.store_column], df[self.config.product_column]]).transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())

                df[f'rolling_std_{window}'] = shifted_values.groupby(
                    [df[self.config.store_column], df[self.config.product_column]]).transform(
                    lambda x: x.rolling(window=window, min_periods=1).std())

                print(f"    Created rolling features with window {window}")

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features from available columns"""
        print("  Creating interaction features...")

        # Price-age interaction (if both available)
        if 'price' in df.columns and 'age' in df.columns:
            df['price_age_interaction'] = df['price'] * df['age']
            print(f"    Created price-age interaction")

        # Store area-amenity interactions (if available)
        if 'store_area' in df.columns and 'has_wifi' in df.columns:
            df['area_wifi_interaction'] = df['store_area'] * df['has_wifi']
            print(f"    Created area-wifi interaction")

        # Time-based interactions
        if 'hour' in df.columns and 'dayofweek' in df.columns:
            df['hour_dayofweek_interaction'] = df['hour'] * df['dayofweek']
            print(f"    Created hour-dayofweek interaction")

        return df

    def _encode_categorical(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Encode categorical column with proper handling of unseen categories"""
        if col not in self.encoders:
            # First time encoding - fit the encoder
            self.encoders[col] = LabelEncoder()
            df_col_filled = df[col].fillna('Unknown').astype(str)
            df[f'{col}_encoded'] = self.encoders[col].fit_transform(df_col_filled)
        else:
            # Transform using existing encoder
            df_col_filled = df[col].fillna('Unknown').astype(str)

            # Handle unseen categories properly
            known_classes = set(self.encoders[col].classes_)
            unseen_categories = set(df_col_filled.unique()) - known_classes

            if unseen_categories:
                print(f"    Warning: Found unseen categories in {col}: {unseen_categories}")
                # Add 'Unknown' to encoder classes if it's not there
                if 'Unknown' not in known_classes:
                    all_categories = list(known_classes) + ['Unknown']
                    self.encoders[col].fit(all_categories)

                # Replace unseen categories with 'Unknown'
                df_col_handled = df_col_filled.copy()
                for unseen_cat in unseen_categories:
                    df_col_handled = df_col_handled.replace(unseen_cat, 'Unknown')
            else:
                df_col_handled = df_col_filled

            try:
                df[f'{col}_encoded'] = self.encoders[col].transform(df_col_handled)
            except ValueError as e:
                print(f"    Error encoding {col}: {e}")
                # Fallback: create new encoder
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df_col_handled)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for modeling - only from available data"""
        exclude_cols = [
            self.config.target_column, self.config.time_column,
            self.config.store_column, self.config.product_column,
            'transaction_id', 'customer_id', 'employee_id', 'payment_method',
            'sales_channel', 'operating_time', 'address', 'acquisition_source',
            'first_purchase', 'feedback_date', 'customer_name', 'product_name',
            'store_name', 'unit_price', 'discount', 'total_amount', 'timestamp'
        ]

        # Exclude original categorical columns that have been encoded
        original_categoricals = []
        for col in df.columns:
            if col.endswith('_encoded'):
                original_col = col.replace('_encoded', '')
                if original_col in df.columns:
                    original_categoricals.append(original_col)

        exclude_cols.extend(original_categoricals)

        feature_cols = [col for col in df.columns if col not in exclude_cols
                        and not col.startswith('id') and df[col].dtype in ['int64', 'float64']]

        print(f"  Selected {len(feature_cols)} feature columns from available data")
        return feature_cols

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for modeling with robust missing value handling"""
        print(f"Preparing data - input shape: {df.shape}")

        # Create features
        df = self.create_features(df)

        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        self.feature_columns = feature_cols

        print(f"Generated {len(feature_cols)} features from available columns")

        # Robust missing value handling
        for col in feature_cols:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                print(f"  Filling {null_count} null values in {col}")

                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)

        # Replace inf values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)

        # Final check for any remaining problematic values
        for col in feature_cols:
            if df[col].isnull().any() or np.isinf(df[col]).any():
                print(f"  WARNING: {col} still has NaN/inf values, setting to 0")
                df[col] = df[col].fillna(0)
                df[col] = df[col].replace([np.inf, -np.inf], 0)

        # Prepare features and target
        X = df[feature_cols].copy()
        y = df[self.config.target_column].copy()

        print(f"Final feature matrix: {X.shape}")
        print(f"Feature columns: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

        # Scale features
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scalers['scaler'].fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scalers['scaler'].transform(X),
                columns=X.columns,
                index=X.index
            )

        return X_scaled, y, feature_cols