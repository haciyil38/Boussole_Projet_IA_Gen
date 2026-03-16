"""
genai_client.py — Client GenAI avec pipeline RAG et cache local
Utilise google-genai (nouveau package).

Installation : pip install google-genai
"""

import os
import json
import hashlib
import time
from pathlib import Path


CACHE_FILE = Path(__file__).parent.parent / "cache" / "genai_cache.json"
CACHE_FILE.parent.mkdir(exist_ok=True)

MIN_CALL_INTERVAL = 2.0
_last_call_time = 0.0


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _make_cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _call_gemini(prompt: str, api_key: str) -> str:
    global _last_call_time

    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        for model_name in ["gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite", "gemini-2.5-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=8192)
                )
                _last_call_time = time.time()
                return response.text
            except Exception as e:
                err = str(e)
                if "404" in err or "not found" in err.lower():
                    continue
                elif "429" in err or "503" in err or "UNAVAILABLE" in err:
                    time.sleep(30)
                    try:
                        response = client.models.generate_content(model=model_name, contents=prompt)
                        _last_call_time = time.time()
                        return response.text
                    except Exception:
                        continue
                else:
                    return f"[Erreur API Gemini : {err}]"

        return "[Erreur API Gemini : aucun modèle disponible]"

    except Exception as e:
        return f"[Erreur API Gemini : {str(e)}]"


def call_with_cache(prompt: str, api_key: str, force_refresh: bool = False) -> tuple[str, bool]:
    cache = _load_cache()
    key = _make_cache_key(prompt)

    if not force_refresh and key in cache:
        return cache[key]["response"], True

    response = _call_gemini(prompt, api_key)

    if not response.startswith("[Erreur"):
        cache[key] = {
            "prompt_hash": key,
            "response": response,
            "timestamp": time.time()
        }
        _save_cache(cache)

    return response, False


# ──────────────────────────────────────────────
# PIPELINE RAG
# ──────────────────────────────────────────────

def build_progression_prompt(
    user_profile: dict,
    top_jobs: list[dict],
    weak_blocs: list[dict],
    gap_analysis: dict
) -> str:
    job_names = [j["titre"] for j in top_jobs[:3]]

    lacunes_text = "\n".join([
        f"- {b['nom']} (score: {b.get('score_utilisateur', b.get('score', 0)):.0%}, gap: {b.get('gap', 0):.0%})"
        for b in gap_analysis.get("lacunes", [])[:5]
    ])

    forces_text = "\n".join([
        f"- {b['nom']} (score: {b['score_utilisateur']:.0%})"
        for b in gap_analysis.get("forces", [])[:3]
    ])

    prompt = f"""Tu es un conseiller en orientation professionnelle expert en Data Science et IA.

PROFIL ANALYSÉ :
- Niveau d'expérience : {user_profile.get('experience', 'Non renseigné')}
- Formation : {user_profile.get('formation', 'Non renseignée')}

RÉSULTATS DE L'ANALYSE SÉMANTIQUE :
- Métiers recommandés (par score d'adéquation) : {', '.join(job_names)}
- Métier cible principal : {job_names[0] if job_names else 'Non déterminé'}

POINTS FORTS IDENTIFIÉS :
{forces_text if forces_text else '- Aucun point fort majeur identifié'}

LACUNES IDENTIFIÉES (à combler en priorité) :
{lacunes_text if lacunes_text else '- Profil globalement adapté au métier cible'}

MISSION :
Génère un plan de progression professionnel personnalisé et actionnable en français.
Le plan doit :
1. Être structuré en 3 phases (court, moyen, long terme : 1 mois, 3 mois, 6 mois)
2. Cibler les lacunes identifiées avec des actions concrètes et des ressources spécifiques
3. Valoriser les points forts existants
4. Être réaliste et motivant
5. Inclure des ressources d'apprentissage concrètes (MOOCs, livres, projets pratiques)
6. Faire environ 300-400 mots

Format : texte structuré avec des titres de phases clairs.
"""
    return prompt


def build_bio_prompt(
    user_profile: dict,
    top_jobs: list[dict],
    block_scores: dict,
    competences_data: dict
) -> str:
    nom_map = {b["id"]: b["nom"] for b in competences_data["blocs"]}

    sorted_blocs = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
    top_blocs = [f"{nom_map.get(bid, bid)} ({score:.0%})" for bid, score in sorted_blocs[:3]]

    job_names = [j["titre"] for j in top_jobs[:2]]

    domaine     = user_profile.get("domaine_nom", "Data & IA")
    prenom      = user_profile.get("prenom", "")
    formation   = user_profile.get("formation", "Bac+5")
    exp         = user_profile.get("experience", "Junior")
    metier_1    = job_names[0] if job_names else "Non déterminé"
    metier_2    = job_names[1] if len(job_names) > 1 else ""
    metiers_str = ", ".join(job_names)
    forces_str  = ", ".join(top_blocs)
    weak_blocs_2 = [
        nom_map.get(bid, bid)
        for bid, score in sorted_blocs[-3:]
        if score < 0.5
    ]

    prompt = f"""Tu es un expert en personal branding et rédaction de profils professionnels LinkedIn, spécialisé dans le secteur {domaine}.

PROFIL COMPLET :
- Prénom : {prenom if prenom else "Non renseigné"}
- Formation : {formation}
- Niveau : {exp}
- Domaine cible : {domaine}
- Métier cible principal : {metier_1}{f" | Métier secondaire : {metier_2}" if metier_2 else ""}
- Compétences fortes (analyse SBERT) : {forces_str}
{f"- Axes de progression identifiés : {', '.join(weak_blocs_2)}" if weak_blocs_2 else ""}

INSTRUCTIONS STRICTES :
Rédige une biographie professionnelle complète et percutante en français, à la 1ère personne (je).

Structure OBLIGATOIRE en 4 paragraphes distincts :

**Paragraphe 1 — Accroche (2 phrases)** : Qui je suis, mon domaine, mon niveau. Phrase d'impact qui donne envie de lire la suite.

**Paragraphe 2 — Compétences techniques (3-4 phrases)** : Détaille mes compétences techniques clés identifiées par l'analyse ({forces_str}). Cite des technologies, outils, méthodes concrètes. Montre la maîtrise technique.

**Paragraphe 3 — Valeur ajoutée & projets (2-3 phrases)** : Ce que j'apporte concrètement. Parle de projets, réalisations, capacité à résoudre des problèmes réels. Lie aux métiers cibles {metiers_str}.

**Paragraphe 4 — Ambition & objectif (2 phrases)** : Mon objectif professionnel clair orienté vers {metier_1}. Phrase de clôture motivante pour un recruteur.

Règles absolues :
- Entre 180 et 250 mots au total
- 1ère personne du singulier (JE) uniquement — jamais "il/elle"
- NE PAS utiliser "je suis passionné(e)" ni "j'adore" — trop vague
- Citer des technologies précises issues du profil (Python, SBERT, RAG, etc.)
- Ton professionnel, dynamique, concret — éviter le langage corporate creux
- Commencer directement par le texte, sans titre ni label de paragraphe
"""
    return prompt


def build_enrichment_prompt(short_text: str, context: str) -> str:
    return f"""Contexte : Un utilisateur répond à un questionnaire d'évaluation de compétences en Data/IA.
Question posée : {context}
Réponse de l'utilisateur (trop courte) : "{short_text}"

Enrichis cette réponse en 2-3 phrases maximum pour qu'elle décrive mieux les compétences techniques.
Reste fidèle au sens original, n'invente pas de compétences non mentionnées.
Réponds uniquement avec la réponse enrichie, sans explication.
"""


def generate_progression_plan(
    user_profile: dict,
    top_jobs: list,
    weak_blocs: list,
    gap_analysis: dict,
    api_key: str
) -> tuple[str, bool]:
    prompt = build_progression_prompt(user_profile, top_jobs, weak_blocs, gap_analysis)
    return call_with_cache(prompt, api_key)


def generate_bio(
    user_profile: dict,
    top_jobs: list,
    block_scores: dict,
    competences_data: dict,
    api_key: str
) -> tuple[str, bool]:
    prompt = build_bio_prompt(user_profile, top_jobs, block_scores, competences_data)
    return call_with_cache(prompt, api_key)


def enrich_short_answer(short_text: str, context: str, api_key: str) -> str:
    prompt = build_enrichment_prompt(short_text, context)
    result, _ = call_with_cache(prompt, api_key)
    return result


def get_cache_stats() -> dict:
    cache = _load_cache()
    return {
        "nb_entrees": len(cache),
        "fichier": str(CACHE_FILE),
        "taille_ko": round(CACHE_FILE.stat().st_size / 1024, 1) if CACHE_FILE.exists() else 0
    }