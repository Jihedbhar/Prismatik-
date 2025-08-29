from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    target_column: str = 'daily_units_sold'
    time_column: str = 'date'
    store_column: str = 'store_id'
    product_column: str = 'product_id'
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    optuna_trials: int = 50
    optuna_timeout: int = 1800
    sequence_length: int = 10
    batch_size: int = 32
    epochs: int = 50


@dataclass
class CSVConfig:
    """CSV file configuration"""
    csv_path: str = 'data/coffee_shop_2.csv'
    # Column mapping from CSV headers to internal names
    column_mapping: Dict[str, str] = None

    def __post_init__(self):
        if self.column_mapping is None:
            self.column_mapping = {
                # Transaction columns
                'id_transaction': 'transaction_id',
                'id_produit': 'product_id',
                'id_magasin': 'store_id',
                'id_client': 'customer_id',
                'id_employé': 'employee_id',
                'date_heure': 'timestamp',
                'quantité': 'quantity',
                'prix_unitaire': 'unit_price',
                'remise': 'discount',
                'montant_total': 'total_amount',
                'méthode_paiement': 'payment_method',

                # Customer columns
                'client_nom': 'customer_name',
                'client_genre': 'gender',
                'client_âge': 'age',
                'client_ville': 'city',
                'client_Tier_fidelité': 'loyalty_tier',
                'client_premier_achat': 'first_purchase',

                # Product columns
                'produit_nom_produit': 'product_name',
                'produit_catégorie': 'category',
                'produit_sous_catégorie': 'subcategory',
                'produit_prix_vente': 'price',

                # Store columns
                'magasin_nom_magasin': 'store_name',
                'magasin_type': 'store_type',
                'magasin_id_localisation': 'location_id'
            }