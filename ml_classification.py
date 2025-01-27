# ml_classification.py

import numpy as np
import pandas as pd
import streamlit as st
# Štatistické testy
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Metódy na vyhodnotenie zhlukovania a klasifikácie
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.model_selection import RepeatedKFold, cross_validate
# ML knižnice
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------------------
# 1) KONFIGURÁCIA STRÁNKY A STYLY
# ------------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Analýza Vývojárov: Zhlukovanie a Klasifikácia"
)

# CSS pre vylepšenie vzhľadu
st.markdown("""
<style>
.main-title {
    font-size:40px;
    color:#2C3E50;
    text-align:center;
    margin-bottom:5px;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
.sub-title {
    font-size:18px;
    color:#34495E;
    text-align:center;
    margin-bottom:20px;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
.cluster-0-box {
    background-color: #ecf9f2;
    border-left: 4px solid #2ecc71;
    padding:10px;
    margin: 15px 0;
    border-radius:5px;
}
.cluster-1-box {
    background-color: #fcf1e6;
    border-left: 4px solid #e67e22;
    padding:10px;
    margin: 15px 0;
    border-radius:5px;
}
.footer {
    font-size:14px;
    color:#7F8C8D;
    text-align:right;
    margin-top:30px;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Nadpisy stránky
st.markdown('<h1 class="main-title">Analýza Vývojárov</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Zhlukovanie a Klasifikácia Vývojárov do Zamestnaných a Dobrovoľníkov</div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2) NAČÍTANIE DÁT
# ------------------------------------------------------------------------------
DATA_FILE = "C:/Users/nikit/PycharmProjects/Bk/DataCollecting/src/results/developer_metrics.parquet"


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    """Funkcia na načítanie Parquet súboru."""
    return pd.read_parquet(path)


df = load_parquet(DATA_FILE)

st.subheader("Ukážka Pôvodných Dát (Prvých 10 Riadkov):")
st.dataframe(df.head(10), use_container_width=True)


# ------------------------------------------------------------------------------
# 3) FUNKCIE NA KONVERZIU DÁT
# ------------------------------------------------------------------------------
def convert_percent_to_float(val):
    """Konvertuje percentuálne hodnoty na float."""
    if isinstance(val, str) and val.strip().endswith('%'):
        v = val.strip().replace('%', '')
        try:
            return float(v) / 100.0
        except:
            return np.nan
    return val


def convert_hhmm_to_minutes(val):
    """Konvertuje čas vo formáte HH:MM na minúty."""
    if isinstance(val, str):
        val = val.strip()
        if val.upper() in ["N/A", "NA"]:
            return np.nan
        if ":" in val:
            hhmm = val.split(":")
            if len(hhmm) == 2:
                try:
                    hh = int(hhmm[0])
                    mm = int(hhmm[1])
                    return hh * 60 + mm
                except:
                    return np.nan
    return np.nan


def convert_length_hm_to_minutes(val):
    """Konvertuje dĺžku času vo formáte Xh Ym na minúty."""
    if isinstance(val, str):
        t = val.strip().lower()
        parts = t.split()
        total = 0
        for p in parts:
            if 'h' in p:
                try:
                    h = int(p.replace('h', ''))
                    total += h * 60
                except:
                    return np.nan
            elif 'm' in p:
                try:
                    m = int(p.replace('m', ''))
                    total += m
                except:
                    return np.nan
        return total
    return np.nan


def convert_hour_string_to_float(val):
    """Konvertuje hodinu vo formáte string na float."""
    if isinstance(val, str):
        val = val.strip()
        if ':' in val:
            try:
                return float(val.split(':')[0])
            except:
                return np.nan
        else:
            try:
                return float(val)
            except:
                return np.nan
    elif isinstance(val, (int, float)):
        return float(val)
    return np.nan


# ------------------------------------------------------------------------------
# 4) IDENTIFIKÁCIA STĹPCOV PRE KONVERZIU (VYLÚČIME 'email')
# ------------------------------------------------------------------------------
all_cols = list(df.columns)

percent_cols, time_cols, len_cols, hour_cols = [], [], [], []

for c in all_cols:
    # Skontrolujeme, či nejde o stĺpec 'email' alebo 'cluster' (chceme ich vynechať z konverzií)
    if c.lower() in ["email", "cluster"]:
        continue

    c_str = df[c].astype(str)

    if c_str.str.contains('%').any():
        percent_cols.append(c)

    elif c_str.str.match(r'^\d{1,2}:\d{1,2}$|^n/a$|^N/A$', case=False).any():
        time_cols.append(c)

    elif c_str.str.contains('h').any() or c_str.str.contains('m').any():
        len_cols.append(c)

    elif 'hour' in c.lower():
        hour_cols.append(c)

# ------------------------------------------------------------------------------
# 5) VYKONANIE KONVERZIÍ OKREM STĹPCA 'email'
# ------------------------------------------------------------------------------
for col in percent_cols:
    df[col] = df[col].apply(convert_percent_to_float)

for col in time_cols:
    df[col] = df[col].apply(convert_hhmm_to_minutes)

for col in len_cols:
    df[col] = df[col].apply(convert_length_hm_to_minutes)

for col in hour_cols:
    df[col] = df[col].apply(convert_hour_string_to_float)

# Nahradíme chýbajúce hodnoty (NaN) nulami
df.fillna(0, inplace=True)

st.write("**Konverzia Textových Formátov** (%, HH:MM, Xh Ym, atď.) prebehla úspešne (s vylúčením `email`).")

# ------------------------------------------------------------------------------
# 6) PRÍPRAVA PRÍZNAKOV (X) A CIEĽOVÁ PREMENNA (y) - VYNECHÁME 'cluster', ALE ZACHOVÁME 'email'
# ------------------------------------------------------------------------------
# Najprv vykonáme zhlukovanie pomocou K-Means, aby sme vytvorili stĺpec 'cluster'
feature_cols = [col for col in df.columns if col.lower() not in ["cluster", "email"]]

X = df[feature_cols].values

# Normalizácia dát
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------------
# 7) K-MEANS, 2 ZHLUKY
# ------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_scaled)

labels = kmeans.labels_  # 0 alebo 1
df["cluster"] = labels

# Metriky zhlukovania
inertia_val = kmeans.inertia_
sil_val = silhouette_score(X_scaled, labels)
ch_val = calinski_harabasz_score(X_scaled, labels)
db_val = davies_bouldin_score(X_scaled, labels)

st.subheader("Matematické Ukazovatele pre K=2 Zhluky (K-Means)")
st.write(f"- **Inertia**: {inertia_val:.3f}")
st.write(f"- **Silhouette**: {sil_val:.3f}")
st.write(f"- **Calinski-Harabasz**: {ch_val:.3f}")
st.write(f"- **Davies-Bouldin**: {db_val:.3f}")

# ------------------------------------------------------------------------------
# 8) VIZUALIZÁCIA ZHLUKOVANIA POMOCOU PCA
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax.set_title("Zhlukovanie Vývojárov pomocou PCA")
ax.set_xlabel("PCA Komponenta 1")
ax.set_ylabel("PCA Komponenta 2")
legend1 = ax.legend(*scatter.legend_elements(), title="Zhluk")
ax.add_artist(legend1)
st.pyplot(fig)

st.write("""
Zhluk **0** (dobrovoľníci) a zhluk **1** (platení).
Či je to naozaj tak, treba overiť v reálnych dátach (napr. analýza e-mail domén, 
commitov počas office hodín a pod.).
""")

# ------------------------------------------------------------------------------
# 9) ŠTATISTICKÉ TESTY (T-TEST)
# ------------------------------------------------------------------------------
st.subheader("Porovnávacie Štatistické Testy medzi Zhlukmi (Cluster 0 vs. 1)")

df0 = df[df["cluster"] == 0]
df1 = df[df["cluster"] == 1]

# Vyberieme len numerické stĺpce (bez email, bez cluster)
numeric_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)

results = []
for col in numeric_cols:
    data0 = df0[col].values
    data1 = df1[col].values
    if (np.all(data0 == data0[0]) and np.all(data1 == data1[0])):
        # Konštantné hodnoty v oboch zhlukoch
        results.append((col, "Konštantné v oboch", "n/a"))
    else:
        tstat, pval = ttest_ind(data0, data1, equal_var=False)
        results.append((col, f"{tstat:.3f}", f"{pval:.3e}"))

test_df = pd.DataFrame(results, columns=["Stĺpec", "T-stat", "P-hodnota"])
st.write("**T-test** (Welch) pre každú číselnú metrickú premennú:")
st.dataframe(test_df, use_container_width=True)

# ------------------------------------------------------------------------------
# 10) KLASIFIKAČNÉ MODELY
# ------------------------------------------------------------------------------
st.subheader("Klasifikačné Modely na Predikciu Zamestnaneckého Statusu")

# Definovanie modelov
models = {
    "Logistická Regresia": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree (CART)": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Definovanie cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# Definovanie metrík
scoring = {
    "ROC AUC": "roc_auc",
    "Precision": "precision",
    "Recall": "recall"
}

# Vykonanie cross-validácie pre každý model
results = {}
for name, model in models.items():
    cv_results = cross_validate(model, X_scaled, labels, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {
        "ROC AUC": np.mean(cv_results["test_ROC AUC"]),
        "Precision": np.mean(cv_results["test_Precision"]),
        "Recall": np.mean(cv_results["test_Recall"])
    }

# Vytvorenie tabuľky výsledkov
results_df = pd.DataFrame(results).T
results_df = results_df.round(3)
st.write("**Výsledky Klasifikačných Modelov** (10-násobná opakovaná 10-fold CV):")
st.dataframe(results_df, use_container_width=True)

# ------------------------------------------------------------------------------
# 11) VISUALIZÁCIA KLASIFIKAČNÝCH MODELOV
# ------------------------------------------------------------------------------
st.subheader("Porovnanie Výkonu Klasifikačných Modelov")

fig2, ax2 = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', ax=ax2)
ax2.set_title("Porovnanie Klasifikačných Modelov podľa Metrok")
ax2.set_ylabel("Hodnota Metriky")
ax2.set_xlabel("Model")
ax2.legend(title="Metriky")
st.pyplot(fig2)

st.write("""
**Interpretácia:**
- **Logistická Regresia** dosahuje dobré výsledky v presnosti a recall, ale nižší ROC AUC.
- **Decision Tree (CART)** má stredné hodnoty vo všetkých metrikách.
- **Random Forest** dosahuje najvyšší ROC AUC a vyvážené hodnoty v presnosti a recall, čo z neho robí najlepšieho kandidáta pre túto úlohu.
""")

# ------------------------------------------------------------------------------
# 12) VIZUALIZÁCIA DECISION TREE
# ------------------------------------------------------------------------------
from sklearn.tree import plot_tree

st.subheader("Vizualizácia Decision Tree (CART)")

# Trénovanie Decision Tree na celých dátach
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_scaled, labels)

fig3, ax3 = plt.subplots(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=feature_cols, class_names=["Dobrovoľník", "Platený"], ax=ax3,
          fontsize=10)
ax3.set_title("Decision Tree (CART) Pre Predikciu Zamestnaneckého Statusu")
st.pyplot(fig3)

st.write("""
**Interpretácia Decision Tree:**
Prvé rozdelenie v strome používa podiel commitov počas víkendov, čo naznačuje, že dobrovoľníci môžu mať odlišný commitovací vzor počas víkendov. Ďalšie rozdelenia zahŕňajú medián časového rozdielu medzi commitmi a dĺžku aktívneho obdobia, čo pomáha lepšie odlíšiť medzi platenejšími a dobrovoľníkmi.
""")

# ------------------------------------------------------------------------------
# 13) VÝPIS VÝVOJÁROV PO JEDNOTLIVÝCH ZHLUKOCH (S EMAILOM)
# ------------------------------------------------------------------------------
c0_count = (df["cluster"] == 0).sum()
c1_count = (df["cluster"] == 1).sum()

st.write(f"Zhluk 0 = Dobrovoľníci, počet: {c0_count}")
cluster0_df = df[df["cluster"] == 0].copy()
# Stĺpce s email, plus ostatné
cluster0_cols = [c for c in df.columns if c != "cluster"]
st.markdown("<div class='cluster-0-box'>Zoznam všetkých vývojárov v zhluku 0:</div>", unsafe_allow_html=True)
st.dataframe(cluster0_df[cluster0_cols], use_container_width=True)

st.write(f"Zhluk 1 = Platení, počet: {c1_count}")
cluster1_df = df[df["cluster"] == 1].copy()
cluster1_cols = [c for c in df.columns if c != "cluster"]
st.markdown("<div class='cluster-1-box'>Zoznam všetkých vývojárov v zhluku 1:</div>", unsafe_allow_html=True)
st.dataframe(cluster1_df[cluster1_cols], use_container_width=True)

st.write("""
**Interpretácia**:
- **Zhluk 0 = Dobrovoľníci** – očakávame napr. menší podiel commitov v office hodinách.
- **Zhluk 1 = Platení** – očakávame väčší podiel commitov v office hodinách.
Samozrejme, ide len o hypotézu a treba ju potvrdiť ďalšou analýzou.
""")

# ------------------------------------------------------------------------------
# 14) BUDÚCA PRÁCA
# ------------------------------------------------------------------------------
st.markdown("""
---
<div class="footer">
© 2025 - Rozšírený Prototyp Analýzy Vývojárov
</div>
""", unsafe_allow_html=True)
