# utils/config.py
import os

# Directories
TEMP_DIR = "temp"
MAPPINGS_DIR = "mappings"

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MAPPINGS_DIR, exist_ok=True)

# Supported database types
DB_TYPES = {
    "SQLite": "sqlite",
    "PostgreSQL": "postgresql",
    "MySQL": "mysql"
}

# Default table and column mappings
DEFAULT_TABLE_MAPPINGS = {
    'Client': 'Client',
    'Employé': 'Employé',
    'Localisation': 'Localisation',
    'Magasin': 'Magasin',
    'Produit': 'Produit',
    'Stock': 'Stock',
    'Transactions': 'Transactions'
}

DEFAULT_COLUMN_MAPPINGS = {
    'Client': {
        'id_client': 'id_client',
        'nom': 'nom',
        'genre': 'genre',
        'âge': 'âge',
        'numéro_téléphone': 'numéro_téléphone',
        'ville': 'ville',
        'CSP': 'CSP',
        'Tier_fidelité': 'Tier_fidelité',
        'premier_achat': 'premier_achat'
    },
    'Employé': {
        'id_employé': 'id_employé',
        'poste': 'poste',
        'id_magasin': 'id_magasin'
    },
   'Localisation': {
        'id_localisation': 'id_localisation',
        'adresse': 'adresse',
        'ville': 'ville',
        'gouvernorat': 'gouvernorat',
        'pays': 'pays'
    },
    'Magasin': {
        'id_magasin': 'id_magasin',
        'nom_magasin': 'nom_magasin',
        'id_localisation': 'id_localisation',
        'type': 'type',
        'horaires_ouverture': 'horaires_ouverture',
        'superficie': 'superficie',
        'wifi': 'wifi',
        'climatisation': 'climatisation',
        'nom_gérant': 'nom_gérant'
    },
    'Produit': {
        'id_produit': 'id_produit',
        'nom_produit': 'nom_produit',
        'description': 'description',
        'catégorie': 'catégorie',
        'sous_catégorie': 'sous_catégorie',
        'prix_achat': 'prix_achat',
        'prix_vente': 'prix_vente'
    },
    'Stock': {
        'id_stock': 'id_stock',
        'id_produit': 'id_produit',
        'id_magasin': 'id_magasin',
        'quantité': 'quantité',
        'seuil_minimum': 'seuil_minimum',
        'dernière_mise_à_jour': 'dernière_mise_à_jour'
    },
    'Transactions': {
        'id_transaction': 'id_transaction',
        'id_produit': 'id_produit',
        'id_magasin': 'id_magasin',
        'id_client': 'id_client',
        'id_employé': 'id_employé',
        'date_heure': 'date_heure',
        'quantité': 'quantité',
        'prix_unitaire': 'prix_unitaire',
        'remise': 'remise',
        'montant_total': 'montant_total',
        'méthode_paiement': 'méthode_paiement'
    }

}