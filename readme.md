# Guide d'installation et de démarrage de l'app 

## Vérification de l'accès à PostgreSQL

Avant de lancer l'application, assurez-vous que votre base de données PostgreSQL est accessible. Voici comment procéder :

### 1. Test de connexion via ligne de commande

Ouvrez un terminal (CMD, PowerShell ou terminal Unix) et utilisez la commande `psql` :

```bash
# Sous Windows
psql -U postgres -h localhost -p 5432 -d retail_platform

# Sous Linux/Mac
psql -U postgres -h localhost -p 5432 -d retail_platform
```

Vous serez invité à saisir le mot de passe de l'utilisateur PostgreSQL. Si la connexion réussit, vous verrez l'invite de commande PostgreSQL :

```
retail_platform=#
```

Pour quitter psql, tapez `\q` puis appuyez sur Entrée.

### 2. Vérification des tables requises

Une fois connecté à PostgreSQL, vous pouvez vérifier que les tables nécessaires existent :

```sql
\dt                           -- Liste toutes les tables
\d nom_de_votre_table_clients -- Affiche la structure de la table clients
\d nom_de_votre_table_produits -- Affiche la structure de la table produits
```

### 3. Résolution des problèmes courants

Si vous ne pouvez pas vous connecter à PostgreSQL :

* **Service PostgreSQL inactif** :

```bash
# Windows
net start postgresql-x64-14   # Remplacez par votre version

# Linux
sudo systemctl start postgresql

# Mac avec Homebrew
brew services start postgresql
```

* **Erreur d'authentification** : Vérifiez que le fichier `pg_hba.conf` autorise les connexions avec mot de passe
* **Port bloqué** : Vérifiez que le pare-feu autorise les connexions sur le port PostgreSQL (5432 par défaut)
* **Base inexistante** : Créez-la si nécessaire

```bash
createdb -U postgres retail_platform
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-organisation/retailoptimizer.git
cd retailoptimizer
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Les dépendances incluent :
* psycopg2 (pour la connexion PostgreSQL)
* pandas
* numpy
* scikit-learn
* tkinter (pour l'interface graphique)
* matplotlib
* seaborn

## Guide de démarrage rapide

### 1. Connexion à votre base de données PostgreSQL

Lancez l'application:

```bash
python main.py
```

L'interface Tkinter s'ouvrira automatiquement avec les valeurs par défaut issues du fichier `db_config.json`. Vous devrez saisir votre mot de passe PostgreSQL et pourrez modifier les autres paramètres si nécessaire.

Cliquez sur "Tester la connexion" pour vérifier que l'application peut accéder à votre base PostgreSQL avant de procéder.

### 2. Mappage des tables et colonnes

Une fois connecté à votre base PostgreSQL, l'application affichera une interface de mappage Tkinter.

Cette interface utilise le schéma défini dans `db_schema.json` pour vous guider dans le mappage de vos tables existantes. Pour chaque table requise:

1. Sélectionnez la table correspondante dans votre base de données PostgreSQL (ex: votre table peut s'appeler "customers" au lieu de "clients")
2. Mappez les colonnes requises avec celles de votre base de données en utilisant l'interface Tkinter
3. Validez le mappage en cliquant sur "Confirmer"

Le système créera un fichier de mappage qui servira de référence pour traduire vos structures de données vers le format attendu par l'application.

Tables requises:
* **clients**: informations sur vos clients
* **produits**: catalogue de vos produits
* **transactions**: historique des ventes
* **stores**: informations sur vos points de vente (si applicable)

### 3. Validation et nettoyage des données

Après le mappage, l'application:
1. Vérifie la conformité des données
2. Nettoie automatiquement les formats et valeurs
3. Affiche un rapport de qualité des données
