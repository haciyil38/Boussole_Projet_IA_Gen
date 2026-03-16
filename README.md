# AISCA — Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences

**Projet certifiant RNCP40875 — Expert en ingénierie de données · Bloc 2**  
EFREI Paris · Mastère Data Engineering & IA · 2025-2026

---

## 🎯 Description

AISCA est un moteur de recommandation sémantique qui :
1. Collecte les compétences d'un utilisateur via un questionnaire hybride
2. Les analyse sémantiquement avec **SBERT** (similarité cosinus)
3. Les compare à un référentiel de 12 blocs de compétences Data/IA
4. Recommande les **Top-3 métiers** les plus adaptés au profil
5. Génère un **plan de progression** et une **bio professionnelle** via un pipeline RAG (Gemini)

---

## 🏗️ Architecture

```
AISCA/
├── app.py                  # Interface Streamlit (point d'entrée)
├── data/
│   ├── competences.json    # Référentiel 12 blocs × 8 compétences
│   └── metiers.json        # 15 profils métiers Data/IA
├── modules/
│   ├── nlp_engine.py       # SBERT + similarité cosinus
│   ├── recommender.py      # Scoring pondéré + Top-N
│   └── genai_client.py     # Pipeline RAG Gemini + cache JSON
├── cache/
│   └── genai_cache.json    # Cache automatique des appels API
├── requirements.txt
└── .env.example
```

## 🔬 Pipeline IA

```
Questionnaire (Streamlit)
    ↓ texte libre + Likert + QCM
Pré-processing (agrégation, enrichissement conditionnel GenAI si < 5 mots)
    ↓
SBERT paraphrase-multilingual-MiniLM-L12-v2 → Embeddings
    ↓
Similarité cosinus vs référentiel de compétences (96 phrases)
    ↓
Score pondéré par bloc : 0.70 × Score_sémantique + 0.30 × Score_Likert
    ↓
Formule globale : Σ(Wi × Si) / Σ(Wi)
    ↓
Top-3 métiers (scoring quadratique sur blocs requis)
    ↓
RAG Gemini (contexte enrichi → 1 appel plan + 1 appel bio)
    ↓
Visualisations (radar + barres + jauges) + tableau de bord
```

## 🚀 Installation et lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-repo/AISCA.git
cd AISCA

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
cp .env.example .env
# Éditez .env et ajoutez votre clé Gemini

# 5. Lancer l'application
streamlit run app.py
```

## 🔑 Obtenir une clé API Gemini (gratuit)

1. Aller sur https://aistudio.google.com
2. Se connecter avec un compte Google
3. Cliquer sur "Get API key"
4. Copier la clé dans votre fichier `.env`

## 📐 Modèle SBERT utilisé

**`paraphrase-multilingual-MiniLM-L12-v2`**
- Multilingue (français + anglais natif)
- Léger : ~120MB, rapide sur CPU
- Optimal pour la similarité de phrases courtes
- Open-source, zéro coût, local

## 🧮 Formule de scoring

**Score par bloc :**
```
Score_bloc = max(0, (mean(max_sim_cosinus) - baseline) / (1 - baseline))
```

**Score final fusionné :**
```
Score_final_bloc = 0.70 × Score_sémantique + 0.30 × Score_Likert
```

**Score global pondéré :**
```
Score_global = Σ(Wi × Si) / Σ(Wi)
```

**Score d'adéquation métier (pondération quadratique) :**
```
Score_métier = Σ(req² × min(1, user/req)) / Σ(req²)
```

## 🛡️ Gouvernance & Éthique

- **Appels API limités** : 2 appels maximum par session (plan + bio)
- **Cache JSON automatique** : réutilisation des réponses identiques
- **Pas de données personnelles stockées** en dehors de la session
- **Transparence** : scores expliqués, sources des recommandations visibles
- **Hallucinations contrôlées** : contexte RAG structuré injecté dans les prompts
- **Modèle local** (SBERT) : zéro appel externe pour la partie NLP cœur

## 📚 Technologies

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Embeddings | SBERT multilingual | Gratuit, local, français |
| Similarité | Cosinus (numpy) | Standard NLP, interprétable |
| Interface | Streamlit | Prototypage rapide, Python natif |
| GenAI | Google Gemini 2.0 Flash | Free Tier, rapide, qualité |
| Visualisation | Plotly | Interactif, professionnel |
| Cache | JSON local | Simple, zero-dependency |
| Versioning | Git / GitHub | Collaboration, traçabilité |
