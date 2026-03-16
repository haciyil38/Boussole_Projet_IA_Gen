"""
nlp_engine.py — Moteur NLP sémantique basé sur SBERT
Calcule les embeddings et la similarité cosinus entre les réponses
utilisateur et le référentiel de compétences.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import os

# Modèle SBERT multilingue (supporte le français et l'anglais)
# Choix justifié : paraphrase-multilingual-MiniLM-L12-v2 est léger,
# gratuit, performant sur le français et adapté à la similarité de phrases.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None  # Singleton pour ne charger le modèle qu'une fois


def get_model() -> SentenceTransformer:
    """Charge le modèle SBERT en mémoire (lazy loading)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_texts(texts: list[str]) -> np.ndarray:
    """
    Encode une liste de textes en vecteurs d'embeddings.
    
    Args:
        texts: Liste de chaînes de caractères à encoder.
    
    Returns:
        Matrice numpy (n_texts x embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de similarité cosinus entre deux ensembles d'embeddings.
    
    Args:
        emb_a: Matrice (n x d)
        emb_b: Matrice (m x d)
    
    Returns:
        Matrice de similarité (n x m), valeurs dans [-1, 1].
    """
    # Normalisation L2
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)
    return np.dot(norm_a, norm_b.T)


def compute_block_score(user_texts: list[str], block_competences: list[str]) -> float:
    """
    Calcule le score de couverture sémantique d'un bloc de compétences
    par rapport aux réponses utilisateur.
    
    Méthode :
      - Pour chaque réponse utilisateur, on prend la similarité maximale
        avec n'importe quelle compétence du bloc (best match).
      - On moyenne ensuite ces maxima sur l'ensemble des réponses.
      - On seuille à 0 pour éviter les scores négatifs.
    
    Args:
        user_texts: Textes libres fournis par l'utilisateur.
        block_competences: Phrases de compétences du référentiel.
    
    Returns:
        Score float dans [0, 1].
    """
    if not user_texts or not block_competences:
        return 0.0

    user_embs = encode_texts(user_texts)
    block_embs = encode_texts(block_competences)

    sim_matrix = cosine_similarity_matrix(user_embs, block_embs)

    # Pour chaque réponse utilisateur, on garde la similarité maximale
    max_sims = sim_matrix.max(axis=1)
    score = float(np.mean(max_sims))

    # Mise à l'échelle : les similarités SBERT sont généralement dans [0.2, 0.9]
    # On renormalise dans [0, 1] avec une baseline minimale de 0.2
    baseline = 0.20
    score = max(0.0, (score - baseline) / (1.0 - baseline))
    return min(1.0, score)


def compute_all_block_scores(
    user_texts: list[str],
    competences_data: dict
) -> dict[str, float]:
    """
    Calcule les scores sémantiques pour tous les blocs du référentiel.
    
    Args:
        user_texts: Réponses textuelles de l'utilisateur (agrégées).
        competences_data: Dictionnaire chargé depuis competences.json.
    
    Returns:
        Dict {bloc_id: score} pour tous les blocs.
    """
    scores = {}
    for bloc in competences_data["blocs"]:
        bloc_id = bloc["id"]
        competences = bloc["competences"]
        score = compute_block_score(user_texts, competences)
        scores[bloc_id] = score
    return scores


def get_weighted_global_score(
    block_scores: dict[str, float],
    competences_data: dict
) -> float:
    """
    Calcule le score global pondéré par les poids des blocs.
    
    Formule : Score_global = Σ(Wi * Si) / Σ(Wi)
    
    Args:
        block_scores: Dict {bloc_id: score}.
        competences_data: Pour récupérer les poids.
    
    Returns:
        Score global dans [0, 1].
    """
    poids = {b["id"]: b["poids"] for b in competences_data["blocs"]}
    
    total_weight = sum(poids[bid] for bid in block_scores)
    weighted_sum = sum(poids[bid] * score for bid, score in block_scores.items())
    
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def get_weakest_blocks(
    block_scores: dict[str, float],
    competences_data: dict,
    n: int = 3
) -> list[dict]:
    """
    Retourne les N blocs avec les scores les plus faibles.
    Utilisé pour le RAG (identification des lacunes à combler).
    
    Args:
        block_scores: Dict {bloc_id: score}.
        competences_data: Pour récupérer les noms des blocs.
        n: Nombre de blocs faibles à retourner.
    
    Returns:
        Liste de dicts {id, nom, score}.
    """
    nom_map = {b["id"]: b["nom"] for b in competences_data["blocs"]}
    sorted_blocs = sorted(block_scores.items(), key=lambda x: x[1])
    return [
        {"id": bid, "nom": nom_map[bid], "score": score}
        for bid, score in sorted_blocs[:n]
    ]
