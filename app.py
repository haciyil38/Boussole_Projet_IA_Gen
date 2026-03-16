"""
app.py — Boussole
"""

import streamlit as st
import json
import os
import sys
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.nlp_engine import compute_all_block_scores, get_weighted_global_score, get_weakest_blocks, get_model
from modules.recommender import get_top_n_recommendations, get_job_gap_analysis
from modules.genai_client import generate_progression_plan, generate_bio, get_cache_stats

# Config
st.set_page_config(page_title="Boussole - oriente vers les métiers, simple et parlant", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_referentiel():
    with open(DATA_DIR / "referentiel.json", encoding="utf-8") as f:
        ref = json.load(f)
    with open(DATA_DIR / "competences_par_domaine.json", encoding="utf-8") as f:
        comp = json.load(f)
    with open(DATA_DIR / "metiers_par_domaine.json", encoding="utf-8") as f:
        met = json.load(f)
    domaines = {d["id"]: d for d in ref["domaines"]}
    return domaines, comp, met

@st.cache_resource
def load_sbert_model():
    return get_model()

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0f2444 0%, #1e4d8c 50%, #0f2444 100%);
    color: white; padding: 2.5rem; border-radius: 16px; text-align: center;
    margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.main-header h1 { font-size: 2.8rem; font-weight: 800; margin: 0; }
.main-header h3 { font-size: 1.1rem; font-weight: 300; margin: 0.5rem 0; opacity: 0.9; }
.main-header p  { font-size: 0.9rem; opacity: 0.7; margin: 0; }
.domain-banner {
    background: linear-gradient(90deg, #1e4d8c11, #3b82f622);
    border: 1.5px solid #3b82f6; border-radius: 12px;
    padding: 0.9rem 1.4rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
}
.metric-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    color: #111111 !important;
}
.metric-card * { color: #111111 !important; }
.metric-card div { color: #111111 !important; }
.score-high { color: #16a34a !important; font-weight: 700; font-size: 2.2rem; }
.score-mid  { color: #d97706 !important; font-weight: 700; font-size: 2.2rem; }
.score-low  { color: #dc2626 !important; font-weight: 700; font-size: 2.2rem; }
.job-card {
    border-left: 5px solid #1e4d8c;
    background: #f0f7ff !important;
    padding: 1rem 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 0.8rem 0;
    color: #111111 !important;
}
.job-card * { color: #111111 !important; }
.job-card h4 { color: #0f2444 !important; font-size: 1.1rem; margin: 0 0 0.4rem 0; }
.job-card p  { color: #333333 !important; margin: 0.2rem 0; }
.job-card-1 { border-left-color: #f59e0b !important; }
.job-card-2 { border-left-color: #94a3b8 !important; }
.job-card-3 { border-left-color: #92400e !important; }
.tag { display: inline-block; background: #e0f2fe; color: #0369a1 !important; border-radius: 20px; padding: 2px 12px; margin: 2px; font-size: 0.82em; font-weight: 500; }
.from-cache { background:#dcfce7; color:#166534 !important; padding:3px 12px; border-radius:20px; font-size:0.8em; }
.from-api   { background:#fef9c3; color:#854d0e !important; padding:3px 12px; border-radius:20px; font-size:0.8em; }
.stProgress > div > div > div > div { background-color: #1e4d8c; }
</style>
""", unsafe_allow_html=True)


# Sidebar

# ── Profils de démonstration ──────────────────────────────────────────
DEMO_PROFILES = {
    "🎓 Data Analyst — Étudiant": {
        "user_profile": {
            "prenom": "Alexandre",
            "formation": "Mastère Data Engineering & IA — EFREI Paris",
            "experience": "Étudiant / Débutant",
            "domaine_id": "DATA_IA",
            "domaine_nom": "Data & Intelligence Artificielle",
            "domaine_icone": "🤖",
        },
        "domaine_id": "DATA_IA",
        "user_responses": [
            "J'ai développé des scripts Python avec Pandas et NumPy pour analyser des données clients. J'utilise Git et GitHub pour collaborer.",
            "J'ai entraîné des modèles Random Forest et XGBoost pour de la classification binaire, évalué avec F1-score et AUC-ROC avec scikit-learn.",
            "J'ai expérimenté avec l'API Gemini pour du prompt engineering et mis en place un pipeline RAG simple pour un projet universitaire.",
            "J'ai conçu des requêtes SQL avancées sur PostgreSQL et manipulé des fichiers CSV et JSON avec Pandas.",
            "Je crée des tableaux de bord avec Streamlit et Plotly, et j'ai réalisé un rapport Power BI lors d'un stage.",
            "Je travaille en méthode Agile Scrum avec Jira et GitHub pour collaborer en binôme.",
            "Langages et frameworks : Python, SQL, FastAPI",
            "Outils ML/DL : scikit-learn, Hugging Face, XGBoost",
            "LLMs / APIs GenAI : Google Gemini, SBERT",
            "Data Engineering / Cloud : PostgreSQL, GCP, Docker",
            "Python / Programmation : niveau 4 sur 5",
            "Mathématiques / Statistiques : niveau 3 sur 5",
            "Traitement des données : niveau 4 sur 5",
            "Machine Learning : niveau 3 sur 5",
            "Deep Learning : niveau 2 sur 5",
            "NLP / Traitement texte : niveau 3 sur 5",
            "IA Générative / LLM : niveau 3 sur 5",
            "Data Engineering : niveau 2 sur 5",
            "Visualisation : niveau 3 sur 5",
            "Gestion de projet : niveau 3 sur 5",
            "Éthique IA / RGPD : niveau 2 sur 5",
            "Cloud / DevOps : niveau 2 sur 5",
        ],
        "likert_scores": {
            "B01": 0.80, "B02": 0.60, "B03": 0.80, "B04": 0.60,
            "B05": 0.40, "B06": 0.60, "B07": 0.60, "B08": 0.40,
            "B09": 0.60, "B10": 0.60, "B11": 0.40, "B12": 0.40,
        },
    },
    "💼 Finance & Banque — Confirmé": {
        "user_profile": {
            "prenom": "Sophie",
            "formation": "Master Finance d'Entreprise — HEC Paris",
            "experience": "Confirmé (2-5 ans)",
            "domaine_id": "FINANCE_BANQUE",
            "domaine_nom": "Finance, Banque & Assurance",
            "domaine_icone": "💰",
        },
        "domaine_id": "FINANCE_BANQUE",
        "user_responses": [
            "J'analyse des états financiers, calcule des ratios de liquidité et de rentabilité, et construis des modèles DCF et LBO sur Excel avancé avec macros VBA.",
            "Je maîtrise la comptabilité générale PCG et les normes IFRS 9 et 16. Je prépare les clôtures mensuelles et annuelles pour un groupe international.",
            "J'ai conduit des analyses de risque crédit et participé à des missions de conformité Bâle III et DORA dans un cabinet d'audit.",
            "J'utilise Bloomberg Terminal pour l'analyse de portefeuilles obligataires et j'ai participé à des opérations de M&A.",
            "J'ai réalisé des missions d'audit légal et des due diligences financières pour des acquisitions de PME.",
            "Je gère la conformité réglementaire KYC, AML et RGPD dans les traitements de données financières sensibles.",
            "Outils financiers : Excel (avancé), Bloomberg, SAP Finance, Power BI, VBA",
            "Normes & Référentiels : IFRS, PCG, Bâle III/IV, AMF",
            "Domaines d'expertise : Comptabilité, Audit, M&A, Contrôle de gestion",
            "Certifications : CFA Level 1, ACCA",
            "Comptabilité générale : niveau 5 sur 5",
            "Analyse financière : niveau 5 sur 5",
            "Gestion des risques : niveau 4 sur 5",
            "Marchés financiers : niveau 4 sur 5",
            "Audit & Contrôle : niveau 4 sur 5",
            "Fiscalité & Droit : niveau 3 sur 5",
            "Assurance & Actuariat : niveau 2 sur 5",
            "Outils financiers : niveau 5 sur 5",
            "Conformité réglementaire : niveau 4 sur 5",
            "Communication financière : niveau 3 sur 5",
        ],
        "likert_scores": {
            "B01": 1.00, "B02": 1.00, "B03": 0.80, "B04": 0.80,
            "B05": 0.80, "B06": 0.60, "B07": 0.40, "B08": 1.00,
            "B09": 0.80, "B10": 0.60,
        },
    },
    "🔬 NLP Engineer — Senior": {
        "user_profile": {
            "prenom": "Karim",
            "formation": "Doctorat en Traitement Automatique des Langues — Sorbonne",
            "experience": "Senior (5+ ans)",
            "domaine_id": "DATA_IA",
            "domaine_nom": "Data & Intelligence Artificielle",
            "domaine_icone": "🤖",
        },
        "domaine_id": "DATA_IA",
        "user_responses": [
            "J'ai développé des bibliothèques Python open-source, conçu des API REST avec FastAPI et utilisé Docker et Kubernetes pour le déploiement en production.",
            "J'ai publié des articles sur les transformers et les architectures BERT, entraîné des modèles sur GPU avec PyTorch et TensorFlow.",
            "Je maîtrise BERT, SBERT, GPT et les pipelines RAG avec LangChain. J'ai déployé des chatbots RAG en production avec Gemini et Claude.",
            "Je conçois des architectures vectorielles avec Pinecone et Chroma, j'orchestre des workflows avec Airflow sur GCP et AWS.",
            "Je crée des visualisations sémantiques avancées et présente mes travaux lors de conférences ACL, EMNLP et NeurIPS.",
            "Je pilote une équipe de 4 ingénieurs NLP en méthode Agile, je rédige des ADR et spécifications techniques.",
            "Langages et frameworks : Python, Scala, SQL, FastAPI, Spark",
            "Outils ML/DL : PyTorch, TensorFlow, Hugging Face, scikit-learn, XGBoost, spaCy",
            "LLMs / APIs GenAI : OpenAI GPT, Google Gemini, Anthropic Claude, LangChain, LlamaIndex, SBERT",
            "Data Engineering / Cloud : PostgreSQL, MongoDB, Kafka, Airflow, AWS, GCP, Docker, Kubernetes",
            "Python / Programmation : niveau 5 sur 5",
            "Mathématiques / Statistiques : niveau 5 sur 5",
            "Traitement des données : niveau 5 sur 5",
            "Machine Learning : niveau 5 sur 5",
            "Deep Learning : niveau 5 sur 5",
            "NLP / Traitement texte : niveau 5 sur 5",
            "IA Générative / LLM : niveau 5 sur 5",
            "Data Engineering : niveau 4 sur 5",
            "Visualisation : niveau 4 sur 5",
            "Gestion de projet : niveau 4 sur 5",
            "Éthique IA / RGPD : niveau 4 sur 5",
            "Cloud / DevOps : niveau 4 sur 5",
        ],
        "likert_scores": {
            "B01": 1.00, "B02": 1.00, "B03": 1.00, "B04": 1.00,
            "B05": 1.00, "B06": 1.00, "B07": 1.00, "B08": 0.80,
            "B09": 0.80, "B10": 0.80, "B11": 0.80, "B12": 0.80,
        },
    },
}


def load_demo_profile(profile_key: str, domaines: dict, competences_par_domaine: dict) -> None:
    """Charge un profil de démo dans le session_state et redirige vers les résultats."""
    p = DEMO_PROFILES[profile_key]
    domaine_id = p["domaine_id"]
    competences_data = competences_par_domaine.get(domaine_id, {})

    # Compléter les likert_scores manquants avec une valeur neutre 0.4
    blocs = competences_data.get("blocs", [])
    likert_full = {b["id"]: 0.4 for b in blocs}
    likert_full.update(p["likert_scores"])

    # Nettoyer les résultats d'une analyse précédente
    for k in ["final_scores", "top_jobs", "weak_blocs", "gap_analysis",
              "global_score", "semantic_scores", "plan_text", "bio_text"]:
        st.session_state.pop(k, None)

    st.session_state.update({
        "user_responses": p["user_responses"],
        "user_profile": p["user_profile"],
        "likert_scores": likert_full,
        "domaine_id": domaine_id,
        "page": "results",
    })

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧭 Boussole")
        st.markdown("**Oriente vers les métiers, simple et parlant**")
        st.markdown("---")
        st.markdown("### Démos instantanées")
        st.caption("Un clic = profil pré-rempli + analyse lancée")
        for label in DEMO_PROFILES:
            if st.button(label, use_container_width=True, key=f"demo_{label}"):
                load_demo_profile(
                    label,
                    st.session_state.get("_domaines_ref", {}),
                    st.session_state.get("_comp_ref", {}),
                )
                st.rerun()
        st.markdown("---")
        st.markdown("### 🔑 Clé API Gemini")
        api_key = st.text_input("Clé Gemini", type="password", value=os.getenv("GEMINI_API_KEY", ""),
                                help="Gratuite sur https://aistudio.google.com")
        if api_key: st.success("✅ Clé configurée")
        else: st.warning("⚠️ Requise pour GenAI")
        st.markdown("---")
        st.markdown("### 📊 Cache GenAI")
        stats = get_cache_stats()
        st.metric("Entrées", stats["nb_entrees"])
        st.metric("Taille", f"{stats['taille_ko']} Ko")
        st.markdown("---")
        st.markdown("### ℹ️ À propos")
        st.markdown("**Boussole** utilise SBERT local + ROME + e-CF 3.0 + Gemini RAG\n\n*Projet certifiant EFREI*")
        return api_key


# Page Questionnaire
def page_questionnaire(domaines, competences_par_domaine):
    st.markdown("""
    <div class="main-header">
        <h1>🧭 Boussole</h1>
        <h3>Oriente vers les métiers, simple et parlant</h3>
        <p>Choisissez votre domaine — les questions s'adaptent automatiquement</p>
    </div>
    """, unsafe_allow_html=True)

    # ÉTAPE 1 : Profil de base (hors formulaire)
    st.header("• Profil général")
    col1, col2, col3 = st.columns(3)
    with col1:
        prenom = st.text_input("Prénom *", placeholder="Ex: Marie", key="prenom_input")
    with col2:
        formation = st.text_input("Formation actuelle ou récente", placeholder="Ex: Master Finance", key="formation_input")
    with col3:
        experience = st.selectbox("Niveau d'expérience *",
            ["Étudiant / Débutant", "Junior (0-2 ans)", "Confirmé (2-5 ans)", "Senior (5+ ans)"],
            key="experience_input")

    st.divider()

    # ÉTAPE 2 : Sélection du domaine (HORS formulaire = rerun immédiat)
    st.header("• Domaine professionnel visé")
    st.caption("Toutes les questions, la grille d'auto-évaluation et les compétences s'adaptent instantanément à votre sélection.")

    domaine_options = list(domaines.values())
    domaine_labels  = [f"{d['icone']}  {d['nom']}" for d in domaine_options]
    domaine_ids     = [d["id"] for d in domaine_options]

    # Sélection en colonnes pour un meilleur visuel
    selected_label = st.selectbox(
        "Sélectionnez votre domaine :",
        domaine_labels,
        key="domaine_select",
        index=domaine_ids.index(st.session_state.get("domaine_id", domaine_ids[0]))
              if st.session_state.get("domaine_id") else 0
    )
    domaine_id  = domaine_ids[domaine_labels.index(selected_label)]
    domaine     = domaines[domaine_id]
    # Stocker immédiatement le domaine dans session_state
    st.session_state["domaine_id"] = domaine_id
    competences_data = competences_par_domaine.get(domaine_id, {})

    # Bannière récapitulative du domaine
    st.markdown(f"""
    <div class="domain-banner">
        <span style="font-size:2rem">{domaine['icone']}</span>
        <div>
            <strong style="font-size:1.1rem">{domaine['nom']}</strong><br>
            <span style="opacity:0.75;font-size:0.9rem">{domaine['description']}</span><br>
            <span style="opacity:0.5;font-size:0.8rem">Source : {domaine['source']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ÉTAPE 3 : Questionnaire dynamique (dans le formulaire)
    st.header(f"{domaine['icone']} Questions personnalisées — {domaine['nom']}")

    with st.form("questionnaire_form", clear_on_submit=False):

        # Questions ouvertes adaptées au domaine
        st.subheader("• Questions ouvertes")
        st.info("💡 Plus vos réponses sont détaillées, plus l'analyse sémantique sera précise.")

        question_responses = {}
        for q in domaine["questions_ouvertes"]:
            question_responses[q["id"]] = st.text_area(
                q["label"], placeholder=q.get("placeholder", ""),
                height=100, key=f"qa_{domaine_id}_{q['id']}"
            )

        st.divider()

        # Auto-évaluation Likert adaptée au domaine
        st.subheader("• Auto-évaluation des compétences clés")
        st.caption(f"Évaluez votre niveau pour chaque compétence liée à **{domaine['nom']}** — 1 = Débutant · 5 = Expert")

        likert_items = domaine["likert_items"]
        likert_scores_raw = {}
        cols_per_row = 3
        rows = [likert_items[i:i+cols_per_row] for i in range(0, len(likert_items), cols_per_row)]
        for row_items in rows:
            cols = st.columns(len(row_items))
            for col, item in zip(cols, row_items):
                with col:
                    likert_scores_raw[item["id"]] = st.slider(
                        item["label"], 1, 5, 2, key=f"lk_{domaine_id}_{item['id']}"
                    )

        st.divider()

        # QCM compétences / outils adaptés au domaine
        st.subheader("• Outils & Compétences techniques")
        qcm_selected = {}
        for section in domaine["qcm_sections"]:
            qcm_selected[section["label"]] = st.multiselect(
                section["label"], section["choices"],
                key=f"qcm_{domaine_id}_{section['label'][:30]}"
            )

        st.divider()
        submitted = st.form_submit_button("🚀 Analyser mon profil", use_container_width=True, type="primary")

    # Traitement après soumission
    if submitted:
        if not prenom:
            st.error("⚠️ Veuillez entrer votre prénom.")
            return

        if not any(v.strip() for v in question_responses.values()):
            st.warning("⚠️ Remplissez au moins une question ouverte pour une analyse précise.")
            return

        # Agrégation des réponses textuelles
        text_responses = []
        for text in question_responses.values():
            if text.strip():
                text_responses.append(text.strip())
        for label, choices in qcm_selected.items():
            if choices:
                text_responses.append(f"{label} : {', '.join(choices)}")
        # Signal Likert en texte
        for item in likert_items:
            bid = item["id"]
            lvl = likert_scores_raw.get(bid, 2)
            text_responses.append(f"{item['label']} : niveau {lvl} sur 5")

        # Normalisation Likert par bloc (0-1)
        blocs = competences_data.get("blocs", [])
        likert_norm = {item["id"]: likert_scores_raw.get(item["id"], 2) / 5.0 for item in likert_items}
        for bloc in blocs:
            if bloc["id"] not in likert_norm:
                likert_norm[bloc["id"]] = 0.4

        st.session_state.update({
            "user_responses": text_responses,
            "user_profile": {
                "prenom": prenom,
                "formation": formation,
                "experience": experience,
                "domaine_id": domaine_id,
                "domaine_nom": domaine["nom"],
                "domaine_icone": domaine["icone"],
            },
            "likert_scores": likert_norm,
            "domaine_id": domaine_id,
            "page": "results"
        })
        st.rerun()


# Page Résultats
def page_results(domaines, competences_par_domaine, metiers_par_domaine, api_key):
    user_profile    = st.session_state.get("user_profile", {})
    user_responses  = st.session_state.get("user_responses", [])
    likert_scores   = st.session_state.get("likert_scores", {})
    domaine_id      = st.session_state.get("domaine_id", "DATA_IA")
 
    prenom     = user_profile.get("prenom", "Utilisateur")
    dom_nom    = user_profile.get("domaine_nom", "")
    dom_icone  = user_profile.get("domaine_icone", "🧠")
 
    competences_data = competences_par_domaine.get(domaine_id, {})
    metiers_data     = metiers_par_domaine.get(domaine_id, {"metiers": []})
 
    st.markdown(f"""
    <div class="main-header">
        <h2>• Résultats — {prenom}</h2>
        <h3>{dom_icone} {dom_nom}</h3>
        <p>Analyse SBERT · Similarité cosinus · Pipeline RAG Gemini</p>
    </div>
    """, unsafe_allow_html=True)
 
    # NLP
    with st.spinner("... Calcul SBERT et similarités cosinus ..."):
        load_sbert_model()
        semantic_scores = compute_all_block_scores(user_responses, competences_data)
        final_scores = {}
        for bid in semantic_scores:
            sem = semantic_scores[bid]
            lik = likert_scores.get(bid, 0.4)
            final_scores[bid] = round(0.70 * sem + 0.30 * lik, 4)
        global_score = get_weighted_global_score(final_scores, competences_data)
        weak_blocs   = get_weakest_blocks(final_scores, competences_data, n=3)
        top_jobs     = get_top_n_recommendations(final_scores, metiers_data, n=3)
        gap_analysis = {}
        if top_jobs and metiers_data["metiers"]:
            best = next((m for m in metiers_data["metiers"] if m["titre"] == top_jobs[0]["titre"]),
                        metiers_data["metiers"][0])
            gap_analysis = get_job_gap_analysis(final_scores, best, competences_data)
 
    st.session_state.update({
        "final_scores": final_scores, "top_jobs": top_jobs,
        "weak_blocs": weak_blocs, "gap_analysis": gap_analysis,
        "global_score": global_score, "semantic_scores": semantic_scores
    })
 
    # Métriques
    st.subheader("• Score global de couverture")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cls = "score-high" if global_score >= 0.65 else ("score-mid" if global_score >= 0.4 else "score-low")
        st.markdown(f'<div class="metric-card"><div class="{cls}">{global_score:.0%}</div><div>Score global</div></div>', unsafe_allow_html=True)
    with c2:
        nb = sum(1 for s in final_scores.values() if s >= 0.6)
        st.markdown(f'<div class="metric-card"><div style="font-size:2.2rem;color:#16a34a;font-weight:700">{nb}</div><div>Blocs ≥ 60%</div></div>', unsafe_allow_html=True)
    with c3:
        if top_jobs:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.05rem;font-weight:700;color:#0f2444">{top_jobs[0]["titre"]}</div><div>Métier n°1</div></div>', unsafe_allow_html=True)
    with c4:
        if top_jobs:
            st.markdown(f'<div class="metric-card"><div style="font-size:2.2rem;color:#1e4d8c;font-weight:700">{top_jobs[0]["pourcentage"]:.0f}%</div><div>Adéquation</div></div>', unsafe_allow_html=True)
 
    st.divider()
 
    # Radar + barres
    nom_map = {b["id"]: b["nom"] for b in competences_data.get("blocs", [])}
    cr, cb2 = st.columns(2)
    with cr:
        st.subheader("• Radar des compétences")
        blocs_k = list(final_scores.keys())
        sc_v = [final_scores[b] for b in blocs_k]
        nm_v = [nom_map.get(b, b) for b in blocs_k]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=sc_v+[sc_v[0]], theta=nm_v+[nm_v[0]], fill="toself",
            fillcolor="rgba(30,77,140,0.2)", line=dict(color="#1e4d8c", width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1], tickformat=".0%")),
            showlegend=False, height=400, margin=dict(l=60,r=60,t=40,b=40))
        st.plotly_chart(fig, use_container_width=True)
    with cb2:
        st.subheader("• Scores par bloc")
        srt = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        nms = [nom_map.get(b,b) for b,_ in srt]
        scs = [s for _,s in srt]
        colors = ["#16a34a" if s>=0.6 else ("#d97706" if s>=0.35 else "#dc2626") for s in scs]
        fig2 = go.Figure(go.Bar(x=scs, y=nms, orientation="h", marker_color=colors,
            text=[f"{s:.0%}" for s in scs], textposition="outside"))
        fig2.update_layout(xaxis=dict(range=[0,1.15], tickformat=".0%"),
            height=400, margin=dict(l=10,r=70,t=20,b=30), plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
 
    st.divider()
 
    # Top-3 métiers
    st.subheader("• Top 3 métiers recommandés")
    medals   = ["🥇","🥈","🥉"]
    card_cls = ["job-card-1","job-card-2","job-card-3"]
    for i, job in enumerate(top_jobs):
        ci, cs = st.columns([3,1])
        with ci:
            tags_s = " ".join([f'<span class="tag">{s}</span>' for s in job.get("secteurs",[])])
            tags_c = " ".join([f'<span class="tag">{c}</span>' for c in job.get("competences_cles",[])])
            st.markdown(f"""
            <div class="job-card {card_cls[i]}">
                <h4>{medals[i]} {job['titre']}</h4>
                <p style="color:#555">{job['description']}</p>
                <p><strong>💰</strong> {job.get('salaire_median','N/A')}</p>
                <p><strong>🏢 Secteurs :</strong> {tags_s}</p>
                <p><strong>🛠️ Compétences :</strong> {tags_c}</p>
            </div>""", unsafe_allow_html=True)
        with cs:
            pct = job["pourcentage"]
            color = "#16a34a" if pct>=65 else ("#d97706" if pct>=40 else "#dc2626")
            st.markdown(f'<div class="metric-card" style="margin-top:1rem"><div style="font-size:2rem;color:{color};font-weight:700">{pct:.0f}%</div><div style="font-size:0.85em;color:#666">Adéquation</div></div>', unsafe_allow_html=True)
            fg = go.Figure(go.Indicator(mode="gauge", value=pct,
                gauge={"axis":{"range":[0,100]},"bar":{"color":color},
                       "steps":[{"range":[0,40],"color":"#fee2e2"},{"range":[40,65],"color":"#fef3c7"},{"range":[65,100],"color":"#dcfce7"}]},
                domain={"x":[0,1],"y":[0,1]}))
            fg.update_layout(height=160, margin=dict(l=15,r=15,t=15,b=15))
            st.plotly_chart(fg, use_container_width=True)
 
    st.divider()
 
    # Forces & Lacunes
    if gap_analysis and top_jobs:
        st.subheader(f"🔍 Analyse détaillée — {top_jobs[0]['titre']}")
        cf, cl = st.columns(2)
        with cf:
            st.markdown("#### ✅ Points forts")
            for f in gap_analysis.get("forces",[])[:5]:
                st.markdown(f"**{f['nom']}**")
                r = min(1.0, f["ratio"])
                st.progress(r, text=f"{r*100:.0f}% du niveau requis")
        with cl:
            st.markdown("#### 📈 Axes de progression")
            for l in gap_analysis.get("lacunes",[])[:5]:
                st.markdown(f"**{l['nom']}**")
                r = min(1.0, l["ratio"])
                st.progress(r, text=f"{r*100:.0f}% du niveau requis")
                if l.get("exemples_competences"):
                    st.caption("À développer : " + " · ".join(l["exemples_competences"][:2]))
 
    st.divider()
 
    # GenAI
    st.subheader("• Génération IA — Rapport personnalisé (RAG + Gemini)")
    if not api_key:
        st.warning("⚠️ Configurez votre clé API Gemini dans la barre latérale.")
    else:
        gp, gb = st.columns(2)
        with gp:
            st.markdown("#### • Plan de progression")
            if st.button("• Générer le plan", use_container_width=True):
                with st.spinner("Génération RAG…"):
                    plan, from_cache = generate_progression_plan(user_profile, top_jobs, weak_blocs, gap_analysis, api_key)
                    st.session_state["plan_text"] = plan
                    st.session_state["plan_from_cache"] = from_cache
            if "plan_text" in st.session_state:
                badge = "from-cache" if st.session_state.get("plan_from_cache") else "from-api"
                label = "• Cache" if st.session_state.get("plan_from_cache") else "• Nouvel appel API"
                st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
                st.markdown("---"); st.markdown(st.session_state["plan_text"])
        with gb:
            st.markdown("#### • Biographie professionnelle")
            if st.button("• Générer la bio", use_container_width=True):
                with st.spinner("Génération RAG…"):
                    bio, from_cache = generate_bio(user_profile, top_jobs, final_scores, competences_data, api_key)
                    st.session_state["bio_text"] = bio
                    st.session_state["bio_from_cache"] = from_cache
            if "bio_text" in st.session_state:
                badge = "from-cache" if st.session_state.get("bio_from_cache") else "from-api"
                label = "• Cache" if st.session_state.get("bio_from_cache") else "• Nouvel appel API"
                st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
                st.markdown("---"); st.markdown(f"> {st.session_state['bio_text']}")
 
    st.divider()
 
    # Tableau détaillé
    with st.expander("• Tableau détaillé des scores par bloc"):
        import pandas as pd
        poids_map = {b["id"]: b["poids"] for b in competences_data.get("blocs", [])}
        df = pd.DataFrame([{
            "Bloc": nom_map.get(bid, bid),
            "Sémantique": f"{semantic_scores.get(bid,0):.1%}",
            "Auto-éval": f"{likert_scores.get(bid,0):.1%}",
            "Score final": f"{s:.1%}",
            "Poids": poids_map.get(bid, 1.0),
            "Niveau": "🟢 Fort" if s>=0.6 else ("🟡 Moyen" if s>=0.35 else "🔴 Faible")
        } for bid, s in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)])
        st.dataframe(df, use_container_width=True, hide_index=True)
 
    st.divider()
    if st.button("← Recommencer avec un nouveau profil", use_container_width=True):
        for k in ["user_responses","user_profile","final_scores","top_jobs","weak_blocs",
                  "gap_analysis","global_score","plan_text","bio_text","likert_scores",
                  "semantic_scores","domaine_id"]:
            st.session_state.pop(k, None)
        st.session_state["page"] = "questionnaire"
        st.rerun()


# Main
def main():
    domaines, competences_par_domaine, metiers_par_domaine = load_referentiel()
    st.session_state["_domaines_ref"] = domaines
    st.session_state["_comp_ref"] = competences_par_domaine
    api_key = render_sidebar()
    if "page" not in st.session_state:
        st.session_state["page"] = "questionnaire"
    if st.session_state["page"] == "questionnaire":
        page_questionnaire(domaines, competences_par_domaine)
    elif st.session_state["page"] == "results":
        page_results(domaines, competences_par_domaine, metiers_par_domaine, api_key)

if __name__ == "__main__":
    main()