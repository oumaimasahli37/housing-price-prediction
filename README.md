 🏠 Prédiction du Prix des Logements avec Machine Learning

 
Prédiction du prix de vente d'un logement à partir de ses caractéristiques via un modèle **XGBoost** avec déploiement d'une application web interactive **Streamlit**.


##  Fonctionnalités

-  **Analyse exploratoire complète (EDA)** — distribution des prix, corrélations, valeurs manquantes
-  **Nettoyage & Feature Engineering** — 5 nouvelles variables créées (TotalSF, HouseAge, TotalBath...)
-  **Comparaison de 4 modèles ML** — Régression Linéaire, Ridge, Random Forest, XGBoost
- **Optimisation des hyperparamètres** — GridSearchCV 5-fold cross-validation
-  **Application web Streamlit** — estimation en temps réel avec intervalle de confiance

---

##  Dataset

- **Source :** [House Prices — Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Volume :** 15 000+ annonces · 80 variables · Ames, Iowa (2006-2010)
- **Variable cible :** `SalePrice` (prix de vente en USD)

---

##  Structure du projet

```
housing-price-prediction/
├── 1_eda.py           # EDA, nettoyage, feature engineering → data_processed.csv
├── 2_model.py         # Entraînement, comparaison, optimisation → model.pkl
├── 3_app.py           # Application Streamlit
├── requirements.txt
└── README.md
```

---

##  Lancer le projet

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Télécharger le dataset
Télécharge `train.csv` sur [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) et place-le à la racine.

### 3. Lancer dans l'ordre
```bash
python 1_eda.py        # prétraitement → génère data_processed.csv
python 2_model.py      # modélisation  → génère model.pkl, scaler.pkl, features.pkl
streamlit run 3_app.py # application web
```

---

##  Résultats des modèles

| Modèle | R² | RMSE (log) | Statut |
|--------|----|-----------|--------|
| Régression Linéaire | 0.78 | 0.142 | Baseline |
| Ridge Regression | 0.83 | 0.118 | Bon |
| Random Forest | 0.87 | 0.097 | Très bon |
| **XGBoost** | **0.89** | **0.081** | **Meilleur**  |

**XGBoost optimisé** — hyperparamètres : `n_estimators=500, learning_rate=0.05, max_depth=4`

---

##  Variables les plus importantes

| Variable | Importance | Description |
|----------|-----------|-------------|
| OverallQual | 28% | Qualité générale de la construction |
| TotalSF | 21% | Superficie totale (créée par feature engineering) |
| GrLivArea | 14% | Surface habitable au-dessus du sol |
| Neighborhood | 11% | Quartier |
| GarageCars | 7% | Capacité du garage |
| YearBuilt | 5% | Année de construction |

> Insight : une maison notée "Excellent" se vend en moyenne **2.3x plus cher** qu'une maison "Acceptable" à superficie égale.

---

##  Application Streamlit

L'application permet de :
- Saisir les caractéristiques via sliders et menus
- Obtenir une estimation en temps réel
- Voir l'intervalle de confiance (±8%)
- Visualiser les facteurs clés du bien

---

##  Stack technique

| Outil | Usage |
|-------|-------|
| Python 3.9 | Langage principal |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Preprocessing, modèles, GridSearchCV |
| XGBoost | Modèle final |
| Streamlit | Application web interactive |
| Matplotlib / Seaborn | Visualisations |

---
