"""
recommender.py — Moteur de recommandation de métiers
Calcule le score d'adéquation de chaque métier au profil utilisateur
et retourne les Top-N recommandations.
"""

import numpy as np


def compute_job_score(
    block_scores: dict[str, float],
    job: dict
) -> float:
    """
    Calcule le score d'adéquation entre le profil utilisateur et un métier.
    
    Méthode :
      Pour chaque bloc requis par le métier, on compare le score utilisateur
      au niveau requis, pondéré par l'importance du bloc dans le métier.
      
      Score_metier = Σ(req_level * min(1, user_score / req_level) * req_level) / Σ(req_level²)
      
      Autrement dit : on pénalise fortement les blocs critiques où l'utilisateur
      est faible, mais on ne pénalise pas pour dépasser le niveau requis.
    
    Args:
        block_scores: Dict {bloc_id: score_utilisateur} dans [0, 1].
        job: Dictionnaire d'un métier depuis metiers.json.
    
    Returns:
        Score d'adéquation dans [0, 1].
    """
    blocs_requis = job.get("blocs_requis", {})
    if not blocs_requis:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for bloc_id, required_level in blocs_requis.items():
        user_score = block_scores.get(bloc_id, 0.0)
        
        # Ratio de couverture (plafonné à 1.0)
        coverage_ratio = min(1.0, user_score / required_level) if required_level > 0 else 1.0
        
        # Pondération quadratique : les blocs très requis pèsent plus
        weight = required_level ** 2
        numerator += coverage_ratio * weight
        denominator += weight

    if denominator == 0:
        return 0.0
    return numerator / denominator


def get_top_n_recommendations(
    block_scores: dict[str, float],
    metiers_data: dict,
    n: int = 3
) -> list[dict]:
    """
    Retourne les N métiers les mieux adaptés au profil utilisateur.
    
    Args:
        block_scores: Scores par bloc du profil utilisateur.
        metiers_data: Dictionnaire chargé depuis metiers.json.
        n: Nombre de recommandations.
    
    Returns:
        Liste triée (desc) de dicts enrichis avec le score d'adéquation.
    """
    scored_jobs = []
    for job in metiers_data["metiers"]:
        score = compute_job_score(block_scores, job)
        job_copy = dict(job)
        job_copy["score_adequation"] = round(score, 4)
        job_copy["pourcentage"] = round(score * 100, 1)
        scored_jobs.append(job_copy)

    # Tri par score décroissant
    scored_jobs.sort(key=lambda x: x["score_adequation"], reverse=True)
    return scored_jobs[:n]


def get_job_gap_analysis(
    block_scores: dict[str, float],
    job: dict,
    competences_data: dict
) -> dict:
    """
    Analyse les écarts entre le profil utilisateur et un métier cible.
    
    Args:
        block_scores: Scores utilisateur par bloc.
        job: Dictionnaire du métier cible.
        competences_data: Pour récupérer les noms des blocs et compétences.
    
    Returns:
        Dict avec les forces, lacunes et score global.
    """
    blocs_requis = job.get("blocs_requis", {})
    nom_map = {b["id"]: b["nom"] for b in competences_data["blocs"]}
    comp_map = {b["id"]: b["competences"] for b in competences_data["blocs"]}

    forces = []
    lacunes = []

    for bloc_id, required_level in blocs_requis.items():
        user_score = block_scores.get(bloc_id, 0.0)
        ratio = user_score / required_level if required_level > 0 else 1.0
        
        bloc_info = {
            "id": bloc_id,
            "nom": nom_map.get(bloc_id, bloc_id),
            "score_utilisateur": round(user_score, 3),
            "niveau_requis": required_level,
            "ratio": round(ratio, 3),
            "gap": round(max(0, required_level - user_score), 3)
        }

        if ratio >= 0.75:
            forces.append(bloc_info)
        else:
            # Ajouter des exemples de compétences à développer
            competences_bloc = comp_map.get(bloc_id, [])
            bloc_info["exemples_competences"] = competences_bloc[:3]
            lacunes.append(bloc_info)

    # Tri par gap décroissant pour les lacunes
    lacunes.sort(key=lambda x: x["gap"], reverse=True)
    forces.sort(key=lambda x: x["ratio"], reverse=True)

    return {
        "metier": job["titre"],
        "score_adequation": compute_job_score(block_scores, job),
        "forces": forces,
        "lacunes": lacunes
    }
