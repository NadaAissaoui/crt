"""
OGiRYS — Matching de Fonctionnalites
Application Streamlit v2.0
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

# ─────────────────────────────────────────────────────────────
#  PARAMETRES FIXES (identiques au notebook)
# ─────────────────────────────────────────────────────────────
BI_NAME   = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CE_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
THRESHOLD = 0.40
TOP_K     = 5
ALPHA     = 0.65

# ─────────────────────────────────────────────────────────────
#  STYLE  — palette violet / rose
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #faf8ff; color: #1a0e2e; }

/* ── Cacher la sidebar completement ── */
section[data-testid="stSidebar"]       { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, #f3e8ff 0%, #fce7f3 50%, #ede9fe 100%);
    border: 1px solid #d8b4fe;
    border-radius: 12px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(168,85,247,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 600;
    color: #7c3aed; margin: 0; letter-spacing: -1px;
}
.hero-sub  { font-size: 0.95rem; color: #9d8ab5; margin-top: 6px; font-weight: 300; }

/* ── Stats ── */
.stats-row { display: flex; gap: 16px; margin-bottom: 24px; }
.stat-card {
    flex: 1; background: #ffffff;
    border: 1px solid #e9d5ff; border-radius: 10px;
    padding: 18px 20px; text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #a855f7; }
.stat-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 600; line-height: 1;
}
.stat-label { font-size: 0.75rem; color: #6d28d9; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-total .stat-number { color: #1a0e2e; }
.stat-oui   .stat-number { color: #7c3aed; }
.stat-non   .stat-number { color: #db2777; }
.stat-score .stat-number { color: #9333ea; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #c026d3) !important;
    color: #ffffff !important;
    border: 1px solid #a855f7 !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    padding: 10px 24px !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #d946ef) !important;
    border-color: #c084fc !important;
    box-shadow: 0 4px 16px rgba(192,132,252,0.35) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #be185d, #9333ea) !important;
    color: #ffffff !important;
    border: 1px solid #ec4899 !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #db2777, #a855f7) !important;
    box-shadow: 0 4px 16px rgba(236,72,153,0.35) !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
    border-radius: 4px !important;
}

/* ── Inputs ── */
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important; border-color: #e9d5ff !important; color: #1a0e2e !important;
}
.stTextInput input {
    background: #ffffff !important; border-color: #e9d5ff !important;
    color: #1a0e2e !important; font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Section title ── */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #7c3aed;
    text-transform: uppercase; letter-spacing: 0.12em;
    border-bottom: 1px solid #e9d5ff;
    padding-bottom: 8px; margin-bottom: 16px;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 8px !important;
    border-left: 3px solid #a855f7 !important;
    background: #faf5ff !important;
}

/* ── Hide streamlit menu ── */
#MainMenu, footer { visibility: hidden; }

/* ── Reponse modele ── */
.model-response {
    background: #faf5ff;
    border: 1px solid #d8b4fe;
    border-left: 4px solid #a855f7;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 16px;
}
.model-response-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
}
.model-response-text {
    font-size: 0.95rem;
    color: #1a0e2e;
    line-height: 1.7;
}
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
        return {
            "etat": "non",
            "commentaire": "Aucune correspondance trouvee.",
            "score": float(hybrid[top_idx[0]]),
            "ce_score": None,
            "mode": "aucun",
            "match": "",
        }

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
        st.error("Colonne 'Fonctionnalite' introuvable dans le fichier Excel.")
        return None
    for col in ["Etat", "Commentaire", "Score_Hybride", "Score_CE", "Match_KB", "Mode"]:
        if col not in excel_df.columns:
            excel_df[col] = ""
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
        excel_df.at[i, "Score_CE"]      = round(r["ce_score"], 4) if r["ce_score"] else None
        excel_df.at[i, "Match_KB"]      = r.get("match", "")
        excel_df.at[i, "Mode"]          = r["mode"]
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
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0


# ─────────────────────────────────────────────────────────────
#  HERO BANNER  — titre unique
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">AO Generator - OGIRYS</div>
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.78rem;color:#9333ea;margin-top:8px;font-weight:300;letter-spacing:0.04em;">Nada's Project — Version 2.0</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  NAVIGATION MANUELLE (remplace st.tabs pour controle complet)
# ─────────────────────────────────────────────────────────────
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0

# CSS pour les onglets custom
st.markdown("""
<style>
.tab-bar { display: flex; gap: 4px; margin-bottom: 24px; border-bottom: 2px solid #e9d5ff; padding-bottom: 0; }
.tab-btn {
    background: transparent; border: none; border-bottom: 3px solid transparent;
    padding: 10px 24px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    font-weight: 600; color: #9d8ab5; cursor: pointer; margin-bottom: -2px;
    transition: all 0.2s;
}
.tab-btn:hover { color: #7c3aed; }
.tab-btn.active { color: #7c3aed; border-bottom: 3px solid #7c3aed; }
</style>
""", unsafe_allow_html=True)

t_col1, t_col2, t_col3 = st.columns([1, 1, 1])
with t_col1:
    if st.button("Upload & Traitement", key="nav_tab0",
                 type="primary" if st.session_state.current_tab == 0 else "secondary",
                 use_container_width=True):
        st.session_state.current_tab = 0
        st.rerun()
with t_col2:
    if st.button("Resultats", key="nav_tab1",
                 type="primary" if st.session_state.current_tab == 1 else "secondary",
                 use_container_width=True):
        st.session_state.current_tab = 1
        st.rerun()
with t_col3:
    if st.button("Recherche manuelle", key="nav_tab2",
                 type="primary" if st.session_state.current_tab == 2 else "secondary",
                 use_container_width=True):
        st.session_state.current_tab = 2
        st.rerun()

st.markdown("<hr style='margin-top:0;border-color:#e9d5ff;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD & TRAITEMENT
# ══════════════════════════════════════════════════════════════
if st.session_state.current_tab == 0:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="section-title">Base de connaissances (draft.csv)</div>', unsafe_allow_html=True)
        kb_file = st.file_uploader(
            "draft.csv", type=["csv"], key="kb_upload",
            label_visibility="collapsed",
        )
        if kb_file:
            try:
                import pandas as _pd_raw
                _df_raw = _pd_raw.read_csv(kb_file, encoding="latin1", nrows=0)
                kb_file.seek(0)
                cols = [c.strip() for c in _df_raw.columns]
                has_ctx  = any("contexte" in c.lower() and "usage" in c.lower() for c in cols)
                has_fonc = any("fonctionnalit" in c.lower() for c in cols)
                has_etat = any("etat" in c.lower() or "état" in c.lower() for c in cols)
                has_comm = any("commentaire" in c.lower() for c in cols)
                if not has_ctx or not has_fonc:
                    manquantes = []
                    if not has_ctx:  manquantes.append("Contexte d'usage")
                    if not has_fonc: manquantes.append("Fonctionnalité")
                    st.session_state.kb_df = None
                    st.error(f"⚠️ Colonnes obligatoires introuvables : **{', '.join(manquantes)}**")
                    st.markdown(
                        "<div style='background:#fdf4ff;border:1px solid #d8b4fe;border-left:4px solid #9333ea;"
                        "border-radius:8px;padding:14px 18px;font-size:0.85rem;color:#3b0764;margin-top:8px;'>"
                        "<b>📋 Colonnes attendues dans draft.csv :</b><br><br>"
                        "&nbsp;&nbsp;✅ <b>Contexte d'usage</b> — obligatoire<br>"
                        "&nbsp;&nbsp;✅ <b>Fonctionnalité</b> — obligatoire<br>"
                        "&nbsp;&nbsp;⚪ <b>Etat</b> — optionnelle (créée si absente)<br>"
                        "&nbsp;&nbsp;⚪ <b>Commentaire</b> — optionnelle (créée si absente)<br><br>"
                        f"<b>Colonnes détectées : {', '.join(cols)}</b></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.kb_df = load_kb(kb_file)
                    if not has_etat:
                        st.session_state.kb_df["Etat"] = ""
                        st.warning("Colonne Etat absente — créée automatiquement.")
                    if not has_comm:
                        st.session_state.kb_df["Commentaire"] = ""
                        st.warning("Colonne Commentaire absente — créée automatiquement.")
                    st.success(f"✅ {len(st.session_state.kb_df)} entrées chargées")
                    with st.expander("Apercu KB"):
                        st.dataframe(st.session_state.kb_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")

    with col_r:
        st.markdown('<div class="section-title">Fichier à traiter (.xlsx)</div>', unsafe_allow_html=True)
        xl_file = st.file_uploader(
            "fichier.xlsx", type=["xlsx", "xls"], key="xl_upload",
            label_visibility="collapsed",
        )
        if xl_file:
            try:
                _df_xl = pd.read_excel(xl_file)
                cols_xl = [c.strip() for c in _df_xl.columns]
                has_fonc_xl = any("fonctionnalit" in c.lower() for c in cols_xl)
                if not has_fonc_xl:
                    st.session_state.excel_df = None
                    st.error("⚠️ Colonne **Fonctionnalité** introuvable dans le fichier Excel.")
                    st.markdown(
                        "<div style='background:#fdf4ff;border:1px solid #d8b4fe;border-left:4px solid #9333ea;"
                        "border-radius:8px;padding:14px 18px;font-size:0.85rem;color:#3b0764;margin-top:8px;'>"
                        "<b>📋 Le fichier Excel doit contenir une colonne dont le nom contient 'Fonctionnalité'.</b><br>"
                        f"Colonnes détectées : {', '.join(cols_xl)}<br><br>"
                        "Vérifiez les noms de colonnes avant de relancer.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.excel_df = _df_xl
                    st.success(f"✅ {len(st.session_state.excel_df)} lignes détectées")
                    with st.expander("Apercu Excel"):
                        st.dataframe(st.session_state.excel_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.divider()

    # ── Chargement modeles + Indexation (bouton unique) ─────────
    st.markdown('<div class="section-title">Initialisation</div>', unsafe_allow_html=True)

    if st.session_state.kb_df is not None:
        if st.button("Charger les modèles & Indexer la KB", key="load_and_index_btn"):
            with st.spinner("Chargement des modèles et calcul des embeddings..."):
                try:
                    bi, ce = load_models()
                    corpus, vec, tmat, emb = build_index(st.session_state.kb_df, bi)
                    st.session_state.vectorizer = vec
                    st.session_state.tfidf_mat  = tmat
                    st.session_state.embeddings = emb
                    st.success(f"Modèles chargés et index construit ({emb.shape[0]} entrées)")
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
        if st.session_state.embeddings is None: missing.append("Index KB (cliquez sur Charger & Indexer)")
        st.info(f"En attente : {' | '.join(missing)}")

    if ready:
        if st.button("Lancer le matching IA", key="run"):
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
                    st.session_state.active_tab = 1
                    progress_bar.progress(1.0, text="Terminé !")
                    oui = (result["Etat"] == "oui").sum()
                    non = (result["Etat"] == "non").sum()
                    st.success(f"✅ Matching terminé ! {oui} présentes / {non} absentes")
                    st.query_params["tab"] = "2"
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")


    # ══════════════════════════════════════════════════════════════
    #  TAB 2 — RESULTATS
    # ══════════════════════════════════════════════════════════════


    # ══════════════════════════════════════════════════════════════
    #  TAB 2 — RESULTATS
    # ══════════════════════════════════════════════════════════════
elif st.session_state.current_tab == 1:
    if st.session_state.result_df is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#6d28d9;">
            <div style="font-size:3rem;margin-bottom:16px;">📊</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;">
                Aucun resultat disponible<br>
                <span style="font-size:0.8rem;opacity:.6;">
                Lancez le traitement dans l'onglet Upload &amp; Traitement</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result_df = st.session_state.result_df
        total  = len(result_df)
        oui    = int((result_df["Etat"] == "oui").sum())
        non    = int((result_df["Etat"] == "non").sum())
        ia_df  = result_df[result_df["Mode"] == "hybride+CE"]
        avg_ce = ia_df["Score_CE"].mean() if len(ia_df) > 0 else 0
        avg_ce = 0 if avg_ce != avg_ce else avg_ce

        # ── Stat cards ────────────────────────────────────────
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card stat-total">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-card stat-oui">
                <div class="stat-number">{oui}</div>
                <div class="stat-label">Presentes</div>
            </div>
            <div class="stat-card stat-non">
                <div class="stat-number">{non}</div>
                <div class="stat-label">Absentes</div>
            </div>
            <div class="stat-card stat-score">
                <div class="stat-number">{avg_ce:.2f}</div>
                <div class="stat-label">Score CE moyen</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Filtres ───────────────────────────────────────────
        st.markdown('<div class="section-title">Filtres</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns([3, 2, 2, 2], gap="small")

        with f1:
            search_q = st.text_input("Rechercher", placeholder="mot-cle...", label_visibility="collapsed")
        with f2:
            etat_filter = st.selectbox("Etat", ["Tous", "oui", "non"], label_visibility="collapsed")
        with f3:
            score_min = st.slider("Score CE min", 0.0, 1.0, 0.0, 0.05, label_visibility="collapsed")
        with f4:
            sort_by = st.selectbox(
                "Trier par",
                ["Score_CE desc", "Score_CE asc", "Score_Hybride desc", "A-Z"],
                label_visibility="collapsed",
            )

        # ── Filtrage ──────────────────────────────────────────
        view     = result_df.copy()
        fonc_col = next((c for c in view.columns if "fonctionnalit" in c.lower()), None)

        if search_q and fonc_col:
            mask = view[fonc_col].str.lower().str.contains(search_q.lower(), na=False)
            if "Match_KB" in view.columns:
                mask |= view["Match_KB"].str.lower().str.contains(search_q.lower(), na=False)
            view = view[mask]

        if etat_filter != "Tous":
            view = view[view["Etat"] == etat_filter]

        if "Score_CE" in view.columns:
            view["Score_CE"] = pd.to_numeric(view["Score_CE"], errors="coerce")
            view = view[view["Score_CE"].fillna(0) >= score_min]

        for num_col in ["Score_CE", "Score_Hybride"]:
            if num_col in view.columns:
                view[num_col] = pd.to_numeric(view[num_col], errors="coerce")

        sort_map = {
            "Score_CE desc":      ("Score_CE", False),
            "Score_CE asc":       ("Score_CE", True),
            "Score_Hybride desc": ("Score_Hybride", False),
            "A-Z":                (fonc_col, True),
        }
        sc, asc_val = sort_map[sort_by]
        if sc and sc in view.columns:
            view = view.sort_values(sc, ascending=asc_val)

        st.caption(f"{len(view)} ligne(s) affichee(s) sur {total}")

        # ── Tableau ───────────────────────────────────────────
        display_cols = [c for c in
            [fonc_col, "Etat", "Score_Hybride", "Score_CE", "Match_KB", "Commentaire", "Mode"]
            if c and c in view.columns]

        st.dataframe(
            view[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=480,
            column_config={
                "Score_CE": st.column_config.ProgressColumn(
                    "Score CE", min_value=0, max_value=1, format="%.3f", width="medium"),
                "Score_Hybride": st.column_config.ProgressColumn(
                    "Score Hybride", min_value=0, max_value=1, format="%.3f", width="medium"),
                "Commentaire": st.column_config.TextColumn("Commentaire", width="large"),
                "Match_KB":    st.column_config.TextColumn("Match KB", width="medium"),
            },
        )

        # ── Download ──────────────────────────────────────────
        cols_to_drop = ["Score_Hybride", "Score_CE", "Match_KB", "Mode"]
        result_dl = result_df.drop(columns=[c for c in cols_to_drop if c in result_df.columns])
        view_dl   = view[display_cols].drop(columns=[c for c in cols_to_drop if c in view[display_cols].columns])

        st.divider()
        dl1, dl2, _ = st.columns([2, 2, 3], gap="medium")
        with dl1:
            st.download_button(
                label="Telecharger Excel complet",
                data=df_to_excel_bytes(result_dl),
                file_name="ogirys_resultats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with dl2:
            st.download_button(
                label="Telecharger vue filtree",
                data=df_to_excel_bytes(view_dl),
                file_name="ogirys_resultats_filtres.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


    # ══════════════════════════════════════════════════════════════
    #  TAB 3 — RECHERCHE MANUELLE
    # ══════════════════════════════════════════════════════════════


    # ══════════════════════════════════════════════════════════════
    #  TAB 3 — RECHERCHE MANUELLE
    # ══════════════════════════════════════════════════════════════
elif st.session_state.current_tab == 2:
    st.markdown('<div class="section-title">Tester une fonctionnalite</div>', unsafe_allow_html=True)

    ready_search = (
        st.session_state.embeddings is not None and
        st.session_state.kb_df is not None
    )

    if not ready_search:
        st.info("Chargez les modeles et indexez la KB dans l'onglet Upload & Traitement.")
    else:
        query = st.text_input(
            "Fonctionnalite a rechercher",
            placeholder="ex: gestion des droits utilisateurs...",
        )
        if st.button("Rechercher", key="manual_search") and query:
            with st.spinner("Recherche en cours..."):
                bi, ce = load_models()
                r = hybrid_search(
                    query, bi, ce,
                    st.session_state.embeddings,
                    st.session_state.vectorizer,
                    st.session_state.tfidf_mat,
                    st.session_state.kb_df,
                )

            is_found   = r["etat"] == "oui"
            label      = "Presente dans OGiRYS." if is_found else "Absente de OGiRYS."
            commentaire = r["commentaire"] if r["commentaire"] else label
            match_text  = r.get("match", "")

            reponse_parts = [label]
            if commentaire and commentaire not in (label, "Pas presente dans OGiRYS."):
                reponse_parts.append(commentaire)
            elif r["etat"] == "non":
                reponse_parts.append("Aucune correspondance suffisante trouvee dans la base de connaissances.")

            reponse_text = "\n".join(reponse_parts)
            reponse_html = "<br>".join(reponse_parts)

            st.markdown(f"""
            <div class="model-response">
                <div class="model-response-title">La réponse du modèle :</div>
                <div class="model-response-text" id="reponse-text">{reponse_html}</div>
                <button onclick="navigator.clipboard.writeText(`{reponse_text}`)"
                    style="margin-top:16px;background:linear-gradient(135deg,#7c3aed,#c026d3);
                    color:#fff;border:none;border-radius:6px;padding:7px 18px;
                    font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                    font-weight:600;cursor:pointer;letter-spacing:0.05em;"
                    onmouseover="this.style.opacity='0.85'"
                    onmouseout="this.style.opacity='1'">
                    ⎘ Copier la réponse
                </button>
            </div>
            """, unsafe_allow_html=True)
