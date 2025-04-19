import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import sqrt
from itertools import product

# === Încarcă datele ===
df = pd.read_csv("Frogs_MFCCs.csv")  # pune numele real aici
X = pd.get_dummies(df.iloc[:, :-1])
y = df.iloc[:, -1]
n_features = X.shape[1]

# === Împărțire train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Parametrii ===
inbag_percents = [0.25, 0.40, 0.60, 0.75, 0.90]
feature_options = [
    'sqrt',                                 # sqrt(n_features)
    int(n_features * 0.10),                 # 10%
    int(n_features * 0.50),                 # 50%
    int(n_features * 0.80),                 # 80%
    int(n_features * 0.90),                 # 90%
    n_features                              # 100%
]

# Convertim numerele la string pt comparabilitate în log
feature_options_str = []
for val in feature_options:
    if val == 'sqrt':
        feature_options_str.append('sqrt')
    else:
        feature_options_str.append(str(max(1, val)))  # min 1 feature

# === Experimente ===
results = []

for inbag, feature in product(inbag_percents, feature_options):
    if feature == 'sqrt':
        max_features = 'sqrt'
    else:
        max_features = max(1, int(feature))

    rf = RandomForestClassifier(
        n_estimators=10,
        max_features=max_features,
        bootstrap=True,
        max_samples=inbag,     # aici setăm procentul in-bag
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        'inbag_percent': int(inbag * 100),
        'max_features': feature if feature == 'sqrt' else int(feature),
        'accuracy': round(acc, 4)
    })

# === Rezultate ===
results_df = pd.DataFrame(results)
print(results_df.sort_values(by='accuracy', ascending=False).reset_index(drop=True))

# === Salvare rezultate în CSV ===
results_df.to_csv("rezultate_random_forest.csv", index=False)
print("\nRezultatele au fost salvate în fisierul 'rezultate_random_forest.csv'.")

import seaborn as sns
import matplotlib.pyplot as plt

# Pregătim datele pentru heatmap
pivot = results_df.pivot(index='inbag_percent', columns='max_features', values='accuracy')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})

plt.title("Acuratețe Random Forest în funcție de % in-bag și număr de feature-uri")
plt.xlabel("Număr de feature-uri testate per split")
plt.ylabel("% in-bag sample")

plt.tight_layout()
plt.savefig("grafic_random_forest.png")  # salvează imaginea
plt.show()
