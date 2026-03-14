# =============================================================================
# 1. EXPLORATION ET PRÉTRAITEMENT DES DONNÉES — PRÉDICTION PRIX LOGEMENTS
# Dataset : Kaggle — House Prices: Advanced Regression Techniques (Ames, Iowa)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  PRÉDICTION PRIX LOGEMENTS — EDA & PRÉTRAITEMENT")
print("=" * 60)

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("\n📂 Chargement des données...")

try:
    df = pd.read_csv('train.csv')
    print(f"✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
except FileNotFoundError:
    print("⚠️  'train.csv' non trouvé.")
    print("    Télécharge-le sur : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques")
    print("    Génération de données simulées...\n")

    np.random.seed(42)
    n = 1460
    neighborhoods = ['NAmes','CollgCr','OldTown','Edwards','Somerst','NridgHt','Gilbert','Sawyer','NWAmes','SawyerW']
    quality       = ['Po','Fa','TA','Gd','Ex']
    quality_map_r = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

    overall_qual  = np.random.choice([3,4,5,6,7,8,9,10], n, p=[0.02,0.08,0.15,0.25,0.25,0.15,0.07,0.03])
    gr_liv_area   = np.random.randint(600, 4000, n)
    year_built    = np.random.randint(1900, 2010, n)
    yr_sold       = np.random.choice([2006,2007,2008,2009,2010], n)
    total_bsmt_sf = np.random.randint(0, 2000, n)
    garage_cars   = np.random.choice([0,1,2,3,4], n, p=[0.05,0.15,0.55,0.20,0.05])
    full_bath     = np.random.choice([0,1,2,3], n, p=[0.02,0.35,0.55,0.08])
    half_bath     = np.random.choice([0,1,2], n, p=[0.50,0.45,0.05])

    # Prix simulé avec logique réaliste
    base_price = (
        overall_qual * 15000 +
        gr_liv_area * 50 +
        (yr_sold - year_built) * -200 +
        total_bsmt_sf * 20 +
        garage_cars * 8000 +
        full_bath * 5000 +
        np.random.normal(0, 20000, n)
    )
    sale_price = np.clip(base_price, 50000, 700000)

    df = pd.DataFrame({
        'Id':            range(1, n+1),
        'OverallQual':   overall_qual,
        'GrLivArea':     gr_liv_area,
        'TotalBsmtSF':   total_bsmt_sf,
        'YearBuilt':     year_built,
        'YrSold':        yr_sold,
        'YearRemodAdd':  year_built + np.random.randint(0, 30, n),
        'GarageCars':    garage_cars,
        'FullBath':      full_bath,
        'HalfBath':      half_bath,
        'BsmtFullBath':  np.random.choice([0,1,2], n, p=[0.40,0.50,0.10]),
        'BsmtHalfBath':  np.random.choice([0,1], n, p=[0.85,0.15]),
        'Neighborhood':  np.random.choice(neighborhoods, n),
        'ExterQual':     np.random.choice(quality, n),
        'KitchenQual':   np.random.choice(quality, n),
        'LotFrontage':   np.where(np.random.rand(n) > 0.18, np.random.randint(20, 130, n), np.nan),
        'Alley':         pd.array([np.random.choice(['Grvl','Pave']) if np.random.rand() > 0.94 else None for _ in range(n)], dtype=object),
        'PoolQC':        pd.array([np.random.choice(['Gd','Ex']) if np.random.rand() > 0.99 else None for _ in range(n)], dtype=object),
        'Fence':         pd.array([np.random.choice(['MnPrv','GdPrv','GdWo','MnWw']) if np.random.rand() > 0.81 else None for _ in range(n)], dtype=object),
        'SalePrice':     sale_price
    })
    print(f"✅ Données simulées : {n:,} lignes")

print(f"\n📊 Aperçu :")
print(df[['OverallQual','GrLivArea','YearBuilt','SalePrice']].head(4).to_string())

# =============================================================================
# 2. ANALYSE DE LA VARIABLE CIBLE
# =============================================================================
print("\n" + "=" * 60)
print("  ANALYSE DE LA VARIABLE CIBLE — SalePrice")
print("=" * 60)

print(f"\n  Min     : ${df['SalePrice'].min():>12,.0f}")
print(f"  Max     : ${df['SalePrice'].max():>12,.0f}")
print(f"  Moyenne : ${df['SalePrice'].mean():>12,.0f}")
print(f"  Médiane : ${df['SalePrice'].median():>12,.0f}")
print(f"  Skewness: {df['SalePrice'].skew():>12.2f}  → transformation log nécessaire")

# Transformation log
df['SalePrice'] = np.log1p(df['SalePrice'])
print(f"\n✅ Transformation log1p appliquée — nouveau skewness : {df['SalePrice'].skew():.2f}")

# =============================================================================
# 3. VALEURS MANQUANTES
# =============================================================================
print("\n" + "=" * 60)
print("  GESTION DES VALEURS MANQUANTES")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\n  Colonnes avec valeurs manquantes : {len(missing)}")
for col, cnt in missing.items():
    pct = cnt / len(df) * 100
    print(f"  {col:<20} : {cnt:>5} ({pct:.1f}%)")

# Imputation numérique : médiane par quartier
if 'LotFrontage' in df.columns:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    print("\n✅ LotFrontage : imputation par médiane du quartier")

# Variables catégorielles → 'None' = absence de fonctionnalité
none_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in none_cols:
    if col in df.columns:
        df[col].fillna('None', inplace=True)

# Restant numérique → 0
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

print("✅ Variables catégorielles absentes → 'None'")
print("✅ Variables numériques restantes → 0")
print(f"✅ Valeurs manquantes restantes : {df.isnull().sum().sum()}")

# =============================================================================
# 4. SUPPRESSION DES OUTLIERS
# =============================================================================
print("\n" + "=" * 60)
print("  SUPPRESSION DES OUTLIERS")
print("=" * 60)

before = len(df)
if 'GrLivArea' in df.columns:
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < np.log1p(200000)))]
print(f"✅ {before - len(df)} outliers supprimés ({len(df):,} lignes restantes)")

# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("  FEATURE ENGINEERING")
print("=" * 60)

# Superficie totale
df['TotalSF']      = df['GrLivArea'] + df.get('TotalBsmtSF', 0)

# Âges
df['HouseAge']     = df['YrSold'] - df['YearBuilt']
df['RemodAge']     = df['YrSold'] - df['YearRemodAdd']
df['IsRemodeled']  = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

# Salles de bain totales
df['TotalBath'] = (
    df.get('FullBath', 0) +
    0.5 * df.get('HalfBath', 0) +
    df.get('BsmtFullBath', 0) +
    0.5 * df.get('BsmtHalfBath', 0)
)

print("✅ TotalSF       = GrLivArea + TotalBsmtSF")
print("✅ HouseAge      = YrSold - YearBuilt")
print("✅ RemodAge      = YrSold - YearRemodAdd")
print("✅ IsRemodeled   = 1 si rénovation effectuée")
print("✅ TotalBath     = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath")

# =============================================================================
# 6. ENCODAGE
# =============================================================================
print("\n" + "=" * 60)
print("  ENCODAGE DES VARIABLES CATÉGORIELLES")
print("=" * 60)

# Encodage ordinal
quality_map = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
for col in ['ExterQual','KitchenQual','PoolQC','FireplaceQu','GarageQual','BsmtQual']:
    if col in df.columns:
        df[col] = df[col].map(quality_map).fillna(0)
print("✅ Variables de qualité → encodage ordinal (0-5)")

# One-Hot Encoding
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"✅ One-Hot Encoding appliqué → {df.shape[1]} colonnes au total")

# =============================================================================
# 7. EXPORT
# =============================================================================
print("\n" + "=" * 60)
print("  EXPORT")
print("=" * 60)

df.to_csv('data_processed.csv', index=False)
print(f"✅ data_processed.csv exporté ({df.shape[0]:,} lignes × {df.shape[1]} colonnes)")
print("\n✅ Étape 1 terminée — Lance maintenant : python 2_model.py")
