"""
OGiRYS — Matching de Fonctionnalites
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="AO Generator - OGIRYS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BI_NAME   = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CE_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
THRESHOLD = 0.40
TOP_K     = 5
ALPHA     = 0.65

# ─────────────────────────────────────────────────────────────
#  STYLE  — charte #482882 / #E2D8F3 / #E8005F
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F7F4FD; color: #1a0e2e; }

section[data-testid="stSidebar"]       { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

.hero-banner {
    background: linear-gradient(135deg, #482882 0%, #6b3fa0 60%, #E8005F 100%);
    border-radius: 14px; padding: 36px 44px;
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: -40%; right: -8%;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(226,216,243,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 600;
    color: #E2D8F3; margin: 0; letter-spacing: -1px;
}
.hero-sub { font-size: 0.82rem; color: rgba(226,216,243,0.75); margin-top: 8px; font-weight: 300; }

.stats-row { display: flex; gap: 16px; margin-bottom: 24px; }
.stat-card {
    flex: 1; background: #ffffff; border: 1px solid #E2D8F3;
    border-radius: 10px; padding: 18px 20px; text-align: center; transition: border-color 0.2s;
}
.stat-card:hover { border-color: #482882; }
.stat-number { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; line-height: 1; }
.stat-label { font-size: 0.75rem; color: #482882; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-total .stat-number { color: #1a0e2e; }
.stat-oui   .stat-number { color: #482882; }
.stat-non   .stat-number { color: #E8005F; }

.stButton > button {
    background: linear-gradient(135deg, #482882, #E8005F) !important;
    color: #ffffff !important; border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    padding: 10px 24px !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; box-shadow: 0 4px 18px rgba(72,40,130,0.3) !important; }
.stButton > button[kind="secondary"] {
    background: #ffffff !important; color: #482882 !important;
    border: 2px solid #E2D8F3 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #482882 !important; background: #F7F4FD !important;
    opacity: 1 !important; box-shadow: none !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #482882, #E8005F) !important;
    color: #ffffff !important; border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 600 !important; width: 100% !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

.stProgress > div > div {
    background: linear-gradient(90deg, #482882, #E8005F) !important; border-radius: 4px !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important; border-color: #E2D8F3 !important; color: #1a0e2e !important;
}
.stTextInput input {
    background: #ffffff !important; border-color: #E2D8F3 !important;
    color: #1a0e2e !important; font-family: 'IBM Plex Mono', monospace !important;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #482882;
    text-transform: uppercase; letter-spacing: 0.12em;
    border-bottom: 2px solid #E2D8F3; padding-bottom: 8px; margin-bottom: 16px;
}
.stAlert { border-radius: 8px !important; border-left: 3px solid #482882 !important; background: #F7F4FD !important; }

.alert-box {
    background: #FFF0F5; border: 1px solid #E8005F;
    border-left: 4px solid #E8005F; border-radius: 8px;
    padding: 14px 18px; font-size: 0.85rem; color: #4a0020; margin-top: 8px;
}
.model-response {
    background: #ffffff; border: 1px solid #E2D8F3;
    border-left: 4px solid #482882; border-radius: 12px; padding: 24px 28px; margin-top: 16px;
}
.model-response-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #482882;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;
}
.model-response-text { font-size: 0.95rem; color: #1a0e2e; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bi = SentenceTransformer(BI_NAME, device=device)
    ce = CrossEncoder(CE_NAME, max_length=512, device=device)
    return bi, ce


# ─────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def load_kb(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="latin1")
    for col in ["Contexte d'usage", "Fonctionnalit\u00e9", "Etat", "Commentaire"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.lower().str.strip()
    return df


def check_cols(df):
    cols = [c.strip() for c in df.columns]
    has_ctx  = any("contexte" in c.lower() and "usage" in c.lower() for c in cols)
    has_fonc = any("fonctionnalit" in c.lower() for c in cols)
    has_etat = any("etat" in c.lower() or "\u00e9tat" in c.lower() for c in cols)
    has_comm = any("commentaire" in c.lower() for c in cols)
    return has_ctx, has_fonc, has_etat, has_comm, cols


def build_index(df, bi_model):
    from sklearn.feature_extraction.text import TfidfVectorizer
    def make_text(r):
        ctx  = r.get("Contexte d'usage", "")
        fonc = r.get("Fonctionnalit\u00e9", "")
        etat = r.get("Etat", "")
        comm = r.get("Commentaire", "")
        return f"{ctx} {fonc} {etat} {comm}"
    corpus     = df.apply(make_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_mat  = vectorizer.fit_transform(corpus)
    embeddings = bi_model.encode(
        corpus, convert_to_numpy=True,
        normalize_embeddings=True, batch_size=32,
        show_progress_bar=False
    )
    return corpus, vectorizer, tfidf_mat, embeddings


def hybrid_search(question, bi_model, ce_model, embeddings, vectorizer, tfidf_mat, df):
    from sklearn.preprocessing import normalize
    q          = str(question).lower().strip()
    emb_q      = bi_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    sem_scores = embeddings @ emb_q
    tfidf_q    = vectorizer.transform([q])
    tfidf_sc   = (normalize(tfidf_mat) @ normalize(tfidf_q).T).toarray().ravel()
    hybrid     = ALPHA * sem_scores + (1 - ALPHA) * tfidf_sc
    top_idx    = hybrid.argsort()[-TOP_K:][::-1]

    if hybrid[top_idx[0]] < THRESHOLD:
        return {"etat": "non", "commentaire": "Aucune correspondance trouvee.",
                "score": float(hybrid[top_idx[0]]), "ce_score": None, "mode": "aucun", "match": ""}

    fonc_col   = next((c for c in df.columns if "fonctionnalit" in c.lower()), df.columns[0])
    candidates = [(i, df.iloc[i][fonc_col], hybrid[i]) for i in top_idx]
    ce_scores  = ce_model.predict([[q, c[1]] for c in candidates], show_progress_bar=False)
    best       = int(np.argmax(ce_scores))
    best_row   = df.iloc[candidates[best][0]]
    is_present = any(w in str(best_row.get("Etat", "")) for w in ["oui", "couvert", "disponible", "present"])

    return {
        "etat":        "oui" if is_present else "non",
        "commentaire": best_row.get("Commentaire", "") if is_present else "Pas presente dans OGiRYS.",
        "score":       float(hybrid[top_idx[0]]),
        "ce_score":    float(ce_scores[best]),
        "mode":        "hybride+CE",
        "match":       best_row.get(fonc_col, ""),
    }


def process_dataframe(excel_df, bi_model, ce_model, embeddings, vectorizer, tfidf_mat, kb_df, progress_cb):
    fonc_col = next((c for c in excel_df.columns if "fonctionnalit" in c.lower()), None)
    if not fonc_col:
        st.error("Colonne 'Fonctionnalité' introuvable dans le fichier Excel.")
        return None

    # 🔥 Création + typage propre des colonnes
    for col in ["Etat", "Commentaire", "Score_Hybride", "Score_CE", "Match_KB", "Mode"]:
        if col not in excel_df.columns:
            if col in ["Score_Hybride", "Score_CE"]:
                excel_df[col] = np.nan
            else:
                excel_df[col] = ""

    # 🔥 FIX PRINCIPAL : forcer les bons types
    excel_df["Etat"] = excel_df["Etat"].astype(str)
    excel_df["Commentaire"] = excel_df["Commentaire"].astype(str)
    excel_df["Match_KB"] = excel_df["Match_KB"].astype(str)
    excel_df["Mode"] = excel_df["Mode"].astype(str)

    excel_df["Score_Hybride"] = pd.to_numeric(excel_df["Score_Hybride"], errors="coerce")
    excel_df["Score_CE"] = pd.to_numeric(excel_df["Score_CE"], errors="coerce")

    total = len(excel_df)

    for i in excel_df.index:
        progress_cb((i + 1) / total)
        fonc = excel_df.at[i, fonc_col]

        if pd.isna(fonc) or str(fonc).strip() == "":
            excel_df.at[i, "Etat"]          = "non"
            excel_df.at[i, "Commentaire"]   = "Cellule vide"
            excel_df.at[i, "Score_Hybride"] = 0.0
            excel_df.at[i, "Mode"]          = "vide"
            continue

        r = hybrid_search(fonc, bi_model, ce_model, embeddings, vectorizer, tfidf_mat, kb_df)

        excel_df.at[i, "Etat"]          = r["etat"]
        excel_df.at[i, "Commentaire"]   = r["commentaire"]
        excel_df.at[i, "Score_Hybride"] = round(r["score"], 4)

        excel_df.at[i, "Score_CE"] = (
            round(r["ce_score"], 4) if r["ce_score"] is not None else np.nan
        )

        excel_df.at[i, "Match_KB"] = r.get("match", "")
        excel_df.at[i, "Mode"]     = r["mode"]

    return excel_df


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
for key in ["kb_df", "excel_df", "result_df", "embeddings", "vectorizer", "tfidf_mat"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0
if "kb_loaded_name" not in st.session_state:
    st.session_state.kb_loaded_name = None
if "xl_loaded_name" not in st.session_state:
    st.session_state.xl_loaded_name = None
if "goto_results" not in st.session_state:
    st.session_state.goto_results = False


# ─────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">AO Generator - OGIRYS</div>
    <div class="hero-sub">Nada's Project \u2014 Version 2.0</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  NAVIGATION (boutons = onglets controlables)
# ─────────────────────────────────────────────────────────────
n1, n2, n3 = st.columns(3)
with n1:
    if st.button("📁  Upload & Traitement", key="nav0",
                 type="primary" if st.session_state.current_tab == 0 else "secondary",
                 use_container_width=True):
        st.session_state.current_tab = 0
        st.rerun()
with n2:
    if st.button("📊  R\u00e9sultats", key="nav1",
                 type="primary" if st.session_state.current_tab == 1 else "secondary",
                 use_container_width=True):
        st.session_state.current_tab = 1
        st.rerun()
with n3:
    if st.button("🔍  Recherche manuelle", key="nav2",
                 type="primary" if st.session_state.current_tab == 2 else "secondary",
                 use_container_width=True):
        st.session_state.current_tab = 2
        st.rerun()

st.markdown("<hr style='margin:0 0 24px 0;border:none;border-top:2px solid #E2D8F3;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ONGLET 1 — UPLOAD & TRAITEMENT
# ══════════════════════════════════════════════════════════════
if st.session_state.current_tab == 0:

    # Redirection automatique apres matching
    if st.session_state.get("goto_results", False):
        st.session_state.goto_results = False
        st.session_state.current_tab = 1
        st.rerun()

    col_l, col_r = st.columns(2, gap="large")

    # ── KB (draft.csv) ────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-title">Base de connaissances (draft.csv)</div>', unsafe_allow_html=True)
        kb_file = st.file_uploader("draft.csv", type=["csv"], key="kb_upload", label_visibility="collapsed")
        if kb_file and kb_file.name != st.session_state.kb_loaded_name:
            try:
                import pandas as _pd_raw
                _df_raw = _pd_raw.read_csv(kb_file, encoding="latin1", nrows=0)
                kb_file.seek(0)
                has_ctx, has_fonc, has_etat, has_comm, cols = check_cols(_df_raw)
                if not has_ctx or not has_fonc:
                    manquantes = []
                    if not has_ctx:  manquantes.append("Contexte d'usage")
                    if not has_fonc: manquantes.append("Fonctionnalit\u00e9")
                    st.session_state.kb_df = None
                    st.session_state.kb_loaded_name = None
                    st.error(f"\u26a0\ufe0f Colonnes obligatoires manquantes : **{', '.join(manquantes)}**")
                    st.markdown(
                        f"<div class='alert-box'><b>\U0001f4cb Colonnes attendues :</b><br><br>"
                        f"&nbsp;&nbsp;\u2705 <b>Contexte d'usage</b> \u2014 obligatoire<br>"
                        f"&nbsp;&nbsp;\u2705 <b>Fonctionnalit\u00e9</b> \u2014 obligatoire<br>"
                        f"&nbsp;&nbsp;\u26aa <b>Etat</b> \u2014 optionnelle<br>"
                        f"&nbsp;&nbsp;\u26aa <b>Commentaire</b> \u2014 optionnelle<br><br>"
                        f"<b>D\u00e9tect\u00e9es : {', '.join(cols)}</b></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.kb_df = load_kb(kb_file)
                    st.session_state.kb_loaded_name = kb_file.name
                    if not has_etat:
                        st.session_state.kb_df["Etat"] = ""
                        st.warning("Colonne **Etat** absente \u2014 cr\u00e9\u00e9e automatiquement.")
                    if not has_comm:
                        st.session_state.kb_df["Commentaire"] = ""
                        st.warning("Colonne **Commentaire** absente \u2014 cr\u00e9\u00e9e automatiquement.")
                    st.success(f"\u2705 {len(st.session_state.kb_df)} entr\u00e9es charg\u00e9es")
                    with st.expander("Aper\u00e7u KB"):
                        st.dataframe(st.session_state.kb_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")
        elif st.session_state.kb_df is not None and kb_file:
            st.success(f"\u2705 {len(st.session_state.kb_df)} entr\u00e9es charg\u00e9es ({kb_file.name})")
       

    # ── Fichier Excel ─────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-title">Fichier \u00e0 traiter (.xlsx)</div>', unsafe_allow_html=True)
        xl_file = st.file_uploader("fichier.xlsx", type=["xlsx", "xls"], key="xl_upload", label_visibility="collapsed")
        if xl_file and xl_file.name != st.session_state.xl_loaded_name:
            try:
                _df_xl = pd.read_excel(xl_file)
                has_ctx_xl, has_fonc_xl, has_etat_xl, has_comm_xl, cols_xl = check_cols(_df_xl)
                if not has_ctx_xl or not has_fonc_xl:
                    manquantes_xl = []
                    if not has_ctx_xl:  manquantes_xl.append("Contexte d'usage")
                    if not has_fonc_xl: manquantes_xl.append("Fonctionnalit\u00e9")
                    st.session_state.excel_df = None
                    st.session_state.xl_loaded_name = None
                    st.error(f"\u26a0\ufe0f Colonnes obligatoires manquantes : **{', '.join(manquantes_xl)}**")
                    st.markdown(
                        f"<div class='alert-box'><b>\U0001f4cb V\u00e9rifiez les noms de colonnes :</b><br><br>"
                        f"&nbsp;&nbsp;\u2705 <b>Contexte d'usage</b> \u2014 obligatoire<br>"
                        f"&nbsp;&nbsp;\u2705 <b>Fonctionnalit\u00e9</b> \u2014 obligatoire<br><br>"
                        f"<b>D\u00e9tect\u00e9es : {', '.join(cols_xl)}</b></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.excel_df = _df_xl
                    st.session_state.xl_loaded_name = xl_file.name
                    st.success(f"\u2705 {len(st.session_state.excel_df)} lignes d\u00e9tect\u00e9es")
                    with st.expander("Aper\u00e7u Excel"):
                        st.dataframe(st.session_state.excel_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")
        elif st.session_state.excel_df is not None and xl_file:
            st.success(f"\u2705 {len(st.session_state.excel_df)} lignes d\u00e9tect\u00e9es ({xl_file.name})")

    st.divider()

    # ── Chargement modeles + Indexation ──────────────────────
    st.markdown('<div class="section-title">Initialisation</div>', unsafe_allow_html=True)
    if st.session_state.kb_df is not None:
        if st.button("Charger les mod\u00e8les & Indexer la KB", key="load_and_index_btn"):
            with st.spinner("Chargement des mod\u00e8les et calcul des embeddings..."):
                try:
                    bi, ce = load_models()
                    corpus, vec, tmat, emb = build_index(st.session_state.kb_df, bi)
                    st.session_state.vectorizer = vec
                    st.session_state.tfidf_mat  = tmat
                    st.session_state.embeddings = emb
                    st.success(f"\u2705 Mod\u00e8les charg\u00e9s \u2014 index construit ({emb.shape[0]} entr\u00e9es)")
                except Exception as e:
                    st.error(f"Erreur : {e}")
    else:
        st.info("Chargez d'abord la base de connaissances (draft.csv) ci-dessus.")

    # ── Lancer le traitement ──────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Traitement</div>', unsafe_allow_html=True)

    ready = all([
        st.session_state.kb_df is not None,
        st.session_state.excel_df is not None,
        st.session_state.embeddings is not None,
    ])

    if not ready:
        missing = []
        if st.session_state.kb_df is None:      missing.append("KB (draft.csv)")
        if st.session_state.excel_df is None:   missing.append("Fichier Excel")
        if st.session_state.embeddings is None: missing.append("Index KB")
        st.info(f"En attente : {' | '.join(missing)}")

    if ready:
        if st.button("\U0001f680  Lancer le matching IA", key="run"):
            progress_bar = st.progress(0, text="Initialisation...")

            def update_progress(pct):
                progress_bar.progress(pct, text=f"Traitement... {int(pct * 100)}%")

            try:
                import copy
                bi, ce     = load_models()
                excel_copy = copy.deepcopy(st.session_state.excel_df)
                result = process_dataframe(
                    excel_copy, bi, ce,
                    st.session_state.embeddings,
                    st.session_state.vectorizer,
                    st.session_state.tfidf_mat,
                    st.session_state.kb_df,
                    update_progress,
                )
                if result is not None:
                    st.session_state.result_df = result
                    st.session_state.goto_results = True
                    progress_bar.progress(1.0, text="Termin\u00e9 !")
                    oui = (result["Etat"] == "oui").sum()
                    non = (result["Etat"] == "non").sum()
                    st.success(f"\u2705 Matching termin\u00e9 ! {oui} pr\u00e9sentes / {non} absentes")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")


# ══════════════════════════════════════════════════════════════
#  ONGLET 2 — RESULTATS
# ══════════════════════════════════════════════════════════════
elif st.session_state.current_tab == 1:

    if st.session_state.result_df is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#482882;">
            <div style="font-size:3rem;margin-bottom:16px;">📊</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;">
                Aucun r\u00e9sultat disponible<br>
                <span style="font-size:0.8rem;opacity:.6;">
                Lancez le traitement dans l'onglet Upload &amp; Traitement</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result_df = st.session_state.result_df
        total = len(result_df)
        oui   = int((result_df["Etat"] == "oui").sum())
        non   = int((result_df["Etat"] == "non").sum())

        # ── Stats (3 cartes, sans score) ─────────────────────
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card stat-total">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-card stat-oui">
                <div class="stat-number">{oui}</div>
                <div class="stat-label">Pr\u00e9sentes</div>
            </div>
            <div class="stat-card stat-non">
                <div class="stat-number">{non}</div>
                <div class="stat-label">Absentes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Filtres ───────────────────────────────────────────
        st.markdown('<div class="section-title">Filtres</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([3, 2, 2], gap="small")
        fonc_col = next((c for c in result_df.columns if "fonctionnalit" in c.lower()), None)

        with f1:
            search_q = st.text_input("Rechercher", placeholder="mot-cl\u00e9...", label_visibility="collapsed")
        with f2:
            etat_filter = st.selectbox("Etat", ["Tous", "oui", "non"], label_visibility="collapsed")
        with f3:
            sort_by = st.selectbox("Trier par", ["A-Z", "Z-A", "Etat"], label_visibility="collapsed")

        # ── Filtrage ──────────────────────────────────────────
        view = result_df.copy()

        if search_q and fonc_col:
            view = view[view[fonc_col].str.lower().str.contains(search_q.lower(), na=False)]
        if etat_filter != "Tous":
            view = view[view["Etat"] == etat_filter]

        sort_map = {"A-Z": (fonc_col, True), "Z-A": (fonc_col, False), "Etat": ("Etat", True)}
        sc, asc_val = sort_map[sort_by]
        if sc and sc in view.columns:
            view = view.sort_values(sc, ascending=asc_val)

        st.caption(f"{len(view)} ligne(s) affich\u00e9e(s) sur {total}")

        # ── Tableau — uniquement Fonctionnalité, Etat, Commentaire
        display_cols = [c for c in [fonc_col, "Etat", "Commentaire"] if c and c in view.columns]

        st.dataframe(
            view[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=500,
            column_config={
                "Etat":        st.column_config.TextColumn("Etat", width="small"),
                "Commentaire": st.column_config.TextColumn("Commentaire", width="large"),
            },
        )

        # ── Download ──────────────────────────────────────────
        st.divider()
        dl1, dl2, _ = st.columns([2, 2, 3], gap="medium")
        with dl1:
            st.download_button(
                label="\u2b07  T\u00e9l\u00e9charger Excel complet",
                data=df_to_excel_bytes(result_df[display_cols]),
                file_name="ogirys_resultats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with dl2:
            st.download_button(
                label="\u2b07  T\u00e9l\u00e9charger vue filtr\u00e9e",
                data=df_to_excel_bytes(view[display_cols]),
                file_name="ogirys_resultats_filtres.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ══════════════════════════════════════════════════════════════
#  ONGLET 3 — RECHERCHE MANUELLE
# ══════════════════════════════════════════════════════════════
elif st.session_state.current_tab == 2:

    st.markdown('<div class="section-title">Tester une fonctionnalit\u00e9</div>', unsafe_allow_html=True)

    ready_search = (
        st.session_state.embeddings is not None and
        st.session_state.kb_df is not None
    )

    if not ready_search:
        st.info("Chargez les mod\u00e8les et indexez la KB dans l'onglet Upload & Traitement.")
    else:
        query = st.text_input(
            "Fonctionnalit\u00e9 \u00e0 rechercher",
            placeholder="ex: gestion des droits utilisateurs...",
        )
        if st.button("\U0001f50d  Rechercher", key="manual_search") and query:
            with st.spinner("Recherche en cours..."):
                bi, ce = load_models()
                r = hybrid_search(
                    query, bi, ce,
                    st.session_state.embeddings,
                    st.session_state.vectorizer,
                    st.session_state.tfidf_mat,
                    st.session_state.kb_df,
                )

            is_found    = r["etat"] == "oui"
            label       = "Pr\u00e9sente dans OGiRYS." if is_found else "Absente de OGiRYS."
            commentaire = r["commentaire"] if r["commentaire"] else label

            reponse_parts = [label]
            if commentaire and commentaire not in (label, "Pas presente dans OGiRYS."):
                reponse_parts.append(commentaire)
            elif not is_found:
                reponse_parts.append("Aucune correspondance suffisante trouv\u00e9e dans la base de connaissances.")

            reponse_text = "\n".join(reponse_parts)
            reponse_html = "<br>".join(reponse_parts)

            st.markdown(f"""
            <div class="model-response">
                <div class="model-response-title">La r\u00e9ponse du mod\u00e8le :</div>
                <div class="model-response-text">{reponse_html}</div>
            </div>
            """, unsafe_allow_html=True)
