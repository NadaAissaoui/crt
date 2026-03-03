"""
OGiRYS — Matching de Fonctionnalites
Application Streamlit v2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import tempfile
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OGiRYS Matching",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  STYLE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d2137 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(33,139,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8b949e;
    margin-top: 6px;
    font-weight: 300;
}
.hero-tag {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 12px;
}

/* ── Stat cards ── */
.stats-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.stat-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #58a6ff; }
.stat-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stat-total  .stat-number { color: #e6edf3; }
.stat-oui    .stat-number { color: #3fb950; }
.stat-non    .stat-number { color: #f85149; }
.stat-score  .stat-number { color: #58a6ff; }

/* ── Upload zones ── */
.upload-zone {
    background: #161b22;
    border: 1.5px dashed #30363d;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #58a6ff; }
.upload-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
    margin-bottom: 8px;
}

/* ── Buttons ── */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    border-color: #3fb950 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(46,160,67,0.3) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: 1px solid #388bfd !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background: #388bfd !important;
    box-shadow: 0 4px 16px rgba(31,111,235,0.35) !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1f6feb, #58a6ff) !important;
    border-radius: 4px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Selectbox / slider ── */
.stSelectbox > div, .stSlider, .stTextInput > div {
    background: #161b22 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.stTextInput input {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Section headers ── */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Badges ── */
.badge-oui {
    background: rgba(63,185,80,0.15);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.4);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-non {
    background: rgba(248,81,73,0.15);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.4);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 8px !important;
    border-left: 3px solid #58a6ff !important;
    background: #161b22 !important;
}

/* ── Hide default streamlit menu ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LAZY MODEL LOADING  (cache = ne charge qu'une fois)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(bi_encoder_name, cross_encoder_name):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bi  = SentenceTransformer(bi_encoder_name, device=device)
    ce  = CrossEncoder(cross_encoder_name, max_length=512, device=device)
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
    corpus = df.apply(
        lambda r: f"{r.get(\"Contexte d'usage\",'')} {r.get('Fonctionnalit\u00e9','')} {r.get('Etat','')} {r.get('Commentaire','')}",
        axis=1
    ).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_mat  = vectorizer.fit_transform(corpus)
    embeddings = bi_model.encode(corpus, convert_to_numpy=True,
                                  normalize_embeddings=True, batch_size=32,
                                  show_progress_bar=False)
    return corpus, vectorizer, tfidf_mat, embeddings


def hybrid_search(question, bi_model, ce_model, embeddings, vectorizer,
                  tfidf_mat, df, threshold, top_k, alpha):
    from sklearn.preprocessing import normalize
    q = str(question).lower().strip()
    emb_q      = bi_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    sem_scores = embeddings @ emb_q
    tfidf_q    = vectorizer.transform([q])
    tfidf_sc   = (normalize(tfidf_mat) @ normalize(tfidf_q).T).toarray().ravel()
    hybrid     = alpha * sem_scores + (1 - alpha) * tfidf_sc
    top_idx    = hybrid.argsort()[-top_k:][::-1]

    if hybrid[top_idx[0]] < threshold:
        return {"etat": "non", "commentaire": "Aucune correspondance trouvee.",
                "score": float(hybrid[top_idx[0]]), "ce_score": None,
                "mode": "aucun", "match": ""}

    fonc_col   = next((c for c in df.columns if "fonctionnalit" in c.lower()), df.columns[0])
    candidates = [(i, df.iloc[i][fonc_col], hybrid[i]) for i in top_idx]
    ce_scores  = ce_model.predict([[q, c[1]] for c in candidates], show_progress_bar=False)
    best       = int(np.argmax(ce_scores))
    best_row   = df.iloc[candidates[best][0]]
    is_present = any(w in str(best_row.get("Etat","")) for w in ["oui","couvert","disponible","present"])

    return {
        "etat":        "oui" if is_present else "non",
        "commentaire": best_row.get("Commentaire","") if is_present else "Pas presente dans OGiRYS.",
        "score":       float(hybrid[top_idx[0]]),
        "ce_score":    float(ce_scores[best]),
        "mode":        "hybride+CE",
        "match":       best_row.get(fonc_col, ""),
    }


def process_dataframe(excel_df, bi_model, ce_model, embeddings, vectorizer,
                      tfidf_mat, kb_df, threshold, top_k, alpha, progress_cb):
    fonc_col = next((c for c in excel_df.columns if "fonctionnalit" in c.lower()), None)
    if not fonc_col:
        st.error("Colonne 'Fonctionnalite' introuvable dans le fichier Excel.")
        return None

    for col in ["Etat","Commentaire","Score_Hybride","Score_CE","Match_KB","Mode"]:
        if col not in excel_df.columns:
            excel_df[col] = ""

    total = len(excel_df)
    for i in excel_df.index:
        progress_cb((i + 1) / total)
        fonc = excel_df.at[i, fonc_col]
        if pd.isna(fonc) or str(fonc).strip() == "":
            excel_df.at[i,"Etat"]="non"; excel_df.at[i,"Commentaire"]="Cellule vide"
            excel_df.at[i,"Score_Hybride"]=0.0; excel_df.at[i,"Mode"]="vide"
            continue
        r = hybrid_search(fonc, bi_model, ce_model, embeddings, vectorizer,
                          tfidf_mat, kb_df, threshold, top_k, alpha)
        excel_df.at[i,"Etat"]          = r["etat"]
        excel_df.at[i,"Commentaire"]   = r["commentaire"]
        excel_df.at[i,"Score_Hybride"] = round(r["score"], 4)
        excel_df.at[i,"Score_CE"]      = round(r["ce_score"], 4) if r["ce_score"] else None
        excel_df.at[i,"Match_KB"]      = r.get("match","")
        excel_df.at[i,"Mode"]          = r["mode"]
    return excel_df


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────
for key in ["kb_df","excel_df","result_df","embeddings","vectorizer","tfidf_mat","bi_model","ce_model"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">Modeles IA</div>', unsafe_allow_html=True)

    bi_name = st.selectbox(
        "Bi-encodeur",
        ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
         "sentence-transformers/all-MiniLM-L6-v2"],
        index=0,
        help="Modele d'embeddings semantiques"
    )
    ce_name = st.selectbox(
        "Cross-Encoder",
        ["cross-encoder/ms-marco-MiniLM-L-6-v2",
         "cross-encoder/ms-marco-MiniLM-L-12-v2"],
        index=0,
        help="Modele de re-ranking"
    )

    st.markdown('<div class="section-title" style="margin-top:24px;">Parametres</div>', unsafe_allow_html=True)

    threshold = st.slider("Seuil de matching", 0.10, 0.90, 0.40, 0.05,
                          help="Score hybride minimum pour valider un match")
    top_k     = st.slider("Top-K candidats (CE)", 1, 10, 5,
                          help="Nombre de candidats passes au cross-encoder")
    alpha     = st.slider("Poids semantique (alpha)", 0.0, 1.0, 0.65, 0.05,
                          help="1.0 = 100% semantique | 0.0 = 100% TF-IDF")

    st.markdown('<div class="section-title" style="margin-top:24px;">A propos</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#8b949e;line-height:1.6;">
    Pipeline IA :<br>
    <span style="color:#58a6ff;">Hybrid Search</span> (TF-IDF + Embeddings)<br>
    + <span style="color:#3fb950;">Cross-Encoder</span> re-ranking<br>
    + <span style="color:#f0883e;">Fine-tuning</span> sur vos donnees
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">OGiRYS Matching</div>
    <div class="hero-sub">Correspondance intelligente de fonctionnalites par IA</div>
    <div>
        <span class="hero-tag">Hybrid Search</span>
        <span class="hero-tag">Cross-Encoder</span>
        <span class="hero-tag">Multilingue</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Upload & Traitement  ", "  Resultats  ", "  Recherche manuelle  "])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD & TRAITEMENT
# ══════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns(2, gap="large")

    # ── Upload KB ─────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-title">Base de connaissances</div>', unsafe_allow_html=True)
        kb_file = st.file_uploader(
            "draft.csv",
            type=["csv"],
            key="kb_upload",
            help="Votre base OGiRYS (colonnes : Contexte d'usage, Fonctionnalite, Etat, Commentaire)",
            label_visibility="collapsed",
        )
        if kb_file:
            try:
                st.session_state.kb_df = load_kb(kb_file)
                st.success(f"{len(st.session_state.kb_df)} entrees chargees")
                with st.expander("Apercu KB"):
                    st.dataframe(st.session_state.kb_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")

    # ── Upload Excel ──────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-title">Fichier a traiter</div>', unsafe_allow_html=True)
        xl_file = st.file_uploader(
            "essaiNadaA.xlsx",
            type=["xlsx","xls"],
            key="xl_upload",
            help="Fichier Excel contenant une colonne 'Fonctionnalite' a remplir",
            label_visibility="collapsed",
        )
        if xl_file:
            try:
                st.session_state.excel_df = pd.read_excel(xl_file)
                st.success(f"{len(st.session_state.excel_df)} lignes detectees")
                with st.expander("Apercu Excel"):
                    st.dataframe(st.session_state.excel_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.divider()

    # ── Chargement des modeles ────────────────────────────────
    st.markdown('<div class="section-title">Initialisation</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([2, 1], gap="medium")
    with col_m1:
        if st.button("Charger les modeles IA", key="load_models"):
            with st.spinner("Chargement des modeles (premiere fois ~2 min)..."):
                try:
                    bi, ce = load_models(bi_name, ce_name)
                    st.session_state.bi_model = bi
                    st.session_state.ce_model = ce
                    st.success("Modeles charges et prets !")
                except Exception as e:
                    st.error(f"Erreur chargement modele : {e}")

    with col_m2:
        if st.session_state.bi_model is not None:
            st.markdown("""
            <div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);
            border-radius:8px;padding:12px;text-align:center;font-family:'IBM Plex Mono',monospace;
            font-size:0.78rem;color:#3fb950;">MODELES PRETS</div>
            """, unsafe_allow_html=True)

    # ── Indexation KB ─────────────────────────────────────────
    if st.session_state.kb_df is not None and st.session_state.bi_model is not None:
        if st.button("Indexer la base de connaissances", key="index_kb"):
            with st.spinner("Calcul des embeddings et TF-IDF..."):
                try:
                    corpus, vec, tmat, emb = build_index(
                        st.session_state.kb_df, st.session_state.bi_model
                    )
                    st.session_state.vectorizer  = vec
                    st.session_state.tfidf_mat   = tmat
                    st.session_state.embeddings  = emb
                    st.success(f"Index construit ({emb.shape[0]} entrees, dim={emb.shape[1]})")
                except Exception as e:
                    st.error(f"Erreur indexation : {e}")

    # ── Lancer le traitement ──────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Traitement</div>', unsafe_allow_html=True)

    ready = all([
        st.session_state.kb_df is not None,
        st.session_state.excel_df is not None,
        st.session_state.bi_model is not None,
        st.session_state.ce_model is not None,
        st.session_state.embeddings is not None,
    ])

    if not ready:
        missing = []
        if st.session_state.kb_df is None:       missing.append("KB (draft.csv)")
        if st.session_state.excel_df is None:    missing.append("Excel a traiter")
        if st.session_state.bi_model is None:    missing.append("Modeles IA")
        if st.session_state.embeddings is None:  missing.append("Index KB")
        st.info(f"En attente : {' | '.join(missing)}")

    if ready:
        if st.button("Lancer le matching IA", key="run"):
            progress_bar = st.progress(0, text="Initialisation...")
            status_text  = st.empty()

            def update_progress(pct):
                progress_bar.progress(pct, text=f"Traitement... {int(pct*100)}%")

            try:
                import copy
                excel_copy = copy.deepcopy(st.session_state.excel_df)
                result = process_dataframe(
                    excel_copy,
                    st.session_state.bi_model,
                    st.session_state.ce_model,
                    st.session_state.embeddings,
                    st.session_state.vectorizer,
                    st.session_state.tfidf_mat,
                    st.session_state.kb_df,
                    threshold, top_k, alpha,
                    update_progress,
                )
                if result is not None:
                    st.session_state.result_df = result
                    progress_bar.progress(1.0, text="Termine !")
                    oui  = (result["Etat"] == "oui").sum()
                    non  = (result["Etat"] == "non").sum()
                    st.success(f"Matching termine ! {oui} presentes / {non} absentes — allez sur l'onglet Resultats")
            except Exception as e:
                st.error(f"Erreur pendant le traitement : {e}")


# ══════════════════════════════════════════════════════════════
#  TAB 2 — RESULTATS
# ══════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.result_df is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#8b949e;">
            <div style="font-size:3rem;margin-bottom:16px;">📊</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;">
                Aucun resultat disponible<br>
                <span style="font-size:0.8rem;opacity:.6;">Lancez le traitement dans l'onglet Upload & Traitement</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result_df = st.session_state.result_df

        # ── Stats cards ───────────────────────────────────────
        total = len(result_df)
        oui   = int((result_df["Etat"] == "oui").sum())
        non   = int((result_df["Etat"] == "non").sum())
        ia_df = result_df[result_df["Mode"] == "hybride+CE"]
        avg_ce = ia_df["Score_CE"].mean() if len(ia_df) > 0 else 0
        avg_ce = 0 if avg_ce != avg_ce else avg_ce

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
            sort_by = st.selectbox("Trier par", ["Score_CE desc", "Score_CE asc", "Score_Hybride desc", "A-Z"],
                                   label_visibility="collapsed")

        # ── Filtrage ──────────────────────────────────────────
        view = result_df.copy()
        fonc_col = next((c for c in view.columns if "fonctionnalit" in c.lower()), None)

        if search_q and fonc_col:
            mask = view[fonc_col].str.lower().str.contains(search_q.lower(), na=False)
            if "Match_KB" in view.columns:
                mask |= view["Match_KB"].str.lower().str.contains(search_q.lower(), na=False)
            view = view[mask]

        if etat_filter != "Tous":
            view = view[view["Etat"] == etat_filter]

        if "Score_CE" in view.columns:
            view = view[view["Score_CE"].fillna(0) >= score_min]

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
                "Etat": st.column_config.TextColumn("Etat", width="small"),
                "Score_CE": st.column_config.ProgressColumn(
                    "Score CE", min_value=0, max_value=1, format="%.3f", width="medium"
                ),
                "Score_Hybride": st.column_config.ProgressColumn(
                    "Score Hybride", min_value=0, max_value=1, format="%.3f", width="medium"
                ),
                "Commentaire": st.column_config.TextColumn("Commentaire", width="large"),
                "Match_KB": st.column_config.TextColumn("Match KB", width="medium"),
            },
        )

        # ── Download ──────────────────────────────────────────
        st.divider()
        dl1, dl2, _ = st.columns([2, 2, 3], gap="medium")
        with dl1:
            st.download_button(
                label="Telecharger Excel complet",
                data=df_to_excel_bytes(result_df),
                file_name="ogirys_resultats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with dl2:
            st.download_button(
                label="Telecharger vue filtree",
                data=df_to_excel_bytes(view[display_cols]),
                file_name="ogirys_resultats_filtres.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ══════════════════════════════════════════════════════════════
#  TAB 3 — RECHERCHE MANUELLE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Tester une fonctionnalite</div>', unsafe_allow_html=True)

    ready_search = all([
        st.session_state.bi_model is not None,
        st.session_state.ce_model is not None,
        st.session_state.embeddings is not None,
        st.session_state.kb_df is not None,
    ])

    if not ready_search:
        st.info("Chargez les modeles et indexez la KB dans l'onglet Upload & Traitement.")
    else:
        query = st.text_input(
            "Fonctionnalite a rechercher",
            placeholder="ex: gestion des droits utilisateurs...",
        )
        if st.button("Rechercher", key="manual_search") and query:
            with st.spinner("Recherche en cours..."):
                r = hybrid_search(
                    query,
                    st.session_state.bi_model,
                    st.session_state.ce_model,
                    st.session_state.embeddings,
                    st.session_state.vectorizer,
                    st.session_state.tfidf_mat,
                    st.session_state.kb_df,
                    threshold, top_k, alpha,
                )

            is_found = r["etat"] == "oui"
            color  = "#3fb950" if is_found else "#f85149"
            label  = "PRESENTE dans OGiRYS" if is_found else "ABSENTE de OGiRYS"
            icon   = "✓" if is_found else "✗"

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {color};border-radius:12px;
                        padding:24px 28px;margin-top:16px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;
                            color:{color};font-weight:600;margin-bottom:16px;">
                    {icon}  {label}
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                    <div>
                        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;
                                    letter-spacing:.08em;margin-bottom:4px;">Score Hybride</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                    color:#58a6ff;">{r['score']:.3f}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;
                                    letter-spacing:.08em;margin-bottom:4px;">Score Cross-Encoder</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                    color:#f0883e;">{f"{r['ce_score']:.3f}" if r['ce_score'] else "N/A"}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;
                                    letter-spacing:.08em;margin-bottom:4px;">Match KB</div>
                        <div style="font-size:0.9rem;color:#e6edf3;">{r.get('match','—')}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;
                                    letter-spacing:.08em;margin-bottom:4px;">Commentaire</div>
                        <div style="font-size:0.9rem;color:#e6edf3;">{r['commentaire']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
