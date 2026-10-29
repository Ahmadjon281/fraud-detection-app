import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

st.set_page_config(page_title="Firibgarlikni aniqlash", layout="wide")
st.title("Firibgarlikni aniqlash: anomaliya modellari")
st.caption("Dataset: Kaggle Credit Card Fraud Detection (sun'iy va anonimlashtirilgan)")

# ---------- 1. Ma'lumot ----------
uploaded = st.file_uploader("creditcard.csv faylini yuklang", type="csv")
if uploaded is None:
    st.info("Davom etish uchun CSV faylni yuklang.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded)
st.subheader("1. Ma'lumot")
st.write(f"Shakl: {df.shape[0]} qator, {df.shape[1]} ustun. NaN: {int(df.isna().sum().sum())}")
st.dataframe(df.head())

# ---------- 2. Tayyorlash ----------
X = df.drop(columns=["Class"]).copy()
y = df["Class"].values
X[["Time", "Amount"]] = RobustScaler().fit_transform(X[["Time", "Amount"]])
X_np = X.astype(np.float32).values
contamination = float(y.mean())
st.write(f"Firibgarlik ulushi: {contamination:.4%}")

# ---------- 3. Modellar ----------
@st.cache_data(show_spinner="Modellar o'rgatilmoqda...")
def run_models(X_np, y, contamination):
    scores, preds = {}, {}

    iso = IsolationForest(n_estimators=100, contamination=contamination,
                          random_state=42, n_jobs=-1).fit(X_np)
    scores["IsolationForest"] = -iso.score_samples(X_np)
    preds["IsolationForest"] = (iso.predict(X_np) == -1).astype(int)

    rng = np.random.RandomState(42)
    idx_lof = rng.choice(len(X_np), 20000, replace=False)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                             contamination=contamination).fit(X_np[idx_lof])
    scores["LOF"] = -lof.score_samples(X_np)
    preds["LOF"] = (lof.predict(X_np) == -1).astype(int)

    ocsvm = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
    ocsvm.fit(X_np[idx_lof[:10000]])
    scores["OneClassSVM"] = -ocsvm.decision_function(X_np)
    preds["OneClassSVM"] = (ocsvm.predict(X_np) == -1).astype(int)

    km = KMeans(n_clusters=8, random_state=42, n_init=3, max_iter=100).fit(X_np)
    dist = np.linalg.norm(X_np - km.cluster_centers_[km.labels_], axis=1)
    scores["KMeans"] = dist
    preds["KMeans"] = (dist > np.percentile(dist, 100 * (1 - contamination))).astype(int)

    return scores, preds

scores, preds = run_models(X_np, y, contamination)

# ---------- 4. Natijalar ----------
st.subheader("2. Baholash natijalari")
rows = []
for name in preds:
    rep = classification_report(y, preds[name], output_dict=True, zero_division=0)["1"]
    rows.append({
        "Model": name,
        "Precision": round(rep["precision"], 4),
        "Recall": round(rep["recall"], 4),
        "ROC-AUC": round(roc_auc_score(y, scores[name]), 4),
        "PR-AUC": round(average_precision_score(y, scores[name]), 4),
    })
results = pd.DataFrame(rows).set_index("Model")
st.dataframe(results)

st.subheader("3. Anomaliya ballari")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, name in zip(axes.flatten(), scores):
    s = pd.Series(scores[name])
    ax.hist(s[y == 0], bins=100, alpha=0.6, density=True, label="Normal", color="steelblue")
    ax.hist(s[y == 1], bins=100, alpha=0.6, density=True, label="Firibgarlik", color="red")
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
st.pyplot(fig)

# ---------- 5. AI tahlil ----------
st.subheader("4. AI tahlil (Gemini)")
if st.button("AI tahlilni yaratish"):
    from google import genai

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    lines = [f"Dataset: {len(y)} ta tranzaksiya, firibgarlik ulushi: {contamination:.4%}",
             results.to_string(), ""]
    for name, pred in preds.items():
        rep = classification_report(y, pred, output_dict=True, zero_division=0)["1"]
        lines.append(f"{name}: aniqlangan={int(pred.sum())}, precision={rep['precision']:.3f}, "
                     f"recall={rep['recall']:.3f}, f1={rep['f1-score']:.3f}")

    prompt = ("Quyida firibgarlikni aniqlash bo'yicha anomaliya modellarining natijalari berilgan.\n\n"
              + "\n".join(lines) +
              "\n\nO'zbek tilida: eng yaxshi model, precision/recall muvozanati, "
              "kuchli va zaif tomonlar, 2-3 ta tavsiya. Hisobot uslubida, qisqa yozing.")

    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    st.markdown(response.text)
    st.caption("Bu matn AI tomonidan yaratilgan. Raqamlarni o'zingiz tekshiring.")
