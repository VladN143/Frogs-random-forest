import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import sqrt
from itertools import product

# === Încarcă datele ===
df = pd.read_csv("Frogs_MFCCs.csv") 

# # === Eliminare outlieri folosind IQR ===
# numeric_cols = df.select_dtypes(include=[np.number]).columns

# Q1 = df[numeric_cols].quantile(0.25)
# Q3 = df[numeric_cols].quantile(0.75)
# IQR = Q3 - Q1

# # Filtrare: păstrăm doar rândurile care NU sunt outlieri
# df_clean = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# print(f"Am eliminat {len(df) - len(df_clean)} outlieri din setul de date.")
# df = df_clean.reset_index(drop=True)

# df = df.drop(columns=['RecordID'])



X = pd.get_dummies(df.iloc[:, :-1])
y = df.iloc[:, -1]
n_features = X.shape[1]



# === Împărțire train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

# # Pregătim datele pentru heatmap
# pivot = results_df.pivot(index='inbag_percent', columns='max_features', values='accuracy')

# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})

# plt.title("Acuratețe Random Forest în funcție de % in-bag și număr de feature-uri")
# plt.xlabel("Număr de feature-uri testate per split")
# plt.ylabel("% in-bag sample")

# plt.tight_layout()
# plt.savefig("grafic_random_forest.png")  # salvează imaginea
# plt.show()

# sns.countplot(x='Genus', data=df)
# plt.title('Distribuția claselor (Genus)')
# plt.xticks(rotation=45)
# plt.show()

# === Încarcă datele
df = pd.read_csv("Frogs_MFCCs.csv")  # pune aici calea corectă

# === Coloana pe care vrei să analizezi outlierii (exemplu: MFCC_1)
coloana = 'Family'  # schimbă cu ce coloană vrei

# === Vizualizare înainte de filtrare
plt.figure(figsize=(12, 5))
sns.boxplot(x=df[coloana])
plt.title(f'Distribuția {coloana} înainte de eliminarea outlierilor')
plt.show()

# === Eliminare outlieri (metoda IQR)
Q1 = df[coloana].quantile(0.25)
Q3 = df[coloana].quantile(0.75)
IQR = Q3 - Q1
limita_min = Q1 - 1.5 * IQR
limita_max = Q3 + 1.5 * IQR

df_fara_outlieri = df[(df[coloana] >= limita_min) & (df[coloana] <= limita_max)]

# === Vizualizare după eliminare
plt.figure(figsize=(12, 5))
sns.boxplot(x=df_fara_outlieri[coloana])
plt.title(f'Distribuția {coloana} după eliminarea outlierilor')
plt.show()

# === Optional: compară numărul de rânduri
print(f"Înainte: {len(df)} înregistrări")
print(f"   După: {len(df_fara_outlieri)} înregistrări")
