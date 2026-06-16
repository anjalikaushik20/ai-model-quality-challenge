"""
Spectral clustering utilities for benchmark item selection.

Design decisions:
  - Gaussian kernel with median-distance heuristic for σ (no tuning needed).
  - SpectralClustering from scikit-learn with precomputed affinity.
  - Two selection criteria per cluster:
        'irt'        → pick item with highest IRT info score (medium stratum)
        'centrality' → pick item closest to cluster centroid (easy/hard strata)
  - Fallback to IRT top-k or even-spread when strata are too small for
    meaningful clustering (n < 2k).
  - Embeddings are computed once and cached to disk, keyed by content hash.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_text_embeddings(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a list of texts using sentence-transformers.

    Caches results to `cache_dir` keyed by an MD5 of the combined text so
    subsequent runs with the same inputs skip the encode step.

    Returns:
        Float32 array of shape (n_texts, embedding_dim).
    """
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = hashlib.md5('|'.join(texts).encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f'emb_{key}.npy'
        if cache_path.exists():
            logger.info(f'Loading embeddings from cache: {cache_path}')
            return np.load(cache_path)
    else:
        cache_path = None

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    logger.info(f'Encoding {len(texts)} texts with {model_name}…')
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32,
        convert_to_numpy=True,
    )

    if cache_path:
        np.save(cache_path, embeddings)
        logger.info(f'Embeddings cached → {cache_path}')

    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    X: np.ndarray,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Gaussian kernel: S_ij = exp(−‖x_i − x_j‖² / (2σ²))

    σ is estimated via the median heuristic when not provided:
        σ = median of all non-zero pairwise L2 distances

    This is a parameter-free, robust choice that adapts to the scale of the
    embedding space.
    """
    diff = X[:, None, :] - X[None, :, :]       # (n, n, d)
    sq_dists = (diff ** 2).sum(axis=2)           # (n, n)

    if sigma is None:
        flat_dists = np.sqrt(sq_dists[sq_dists > 1e-12])
        sigma = float(np.median(flat_dists)) if len(flat_dists) > 0 else 1.0
        sigma = max(sigma, 1e-8)

    S = np.exp(-sq_dists / (2.0 * sigma ** 2))
    return S.astype(np.float32)


# ---------------------------------------------------------------------------
# Cluster-based selection
# ---------------------------------------------------------------------------

def spectral_select(
    X: np.ndarray,
    k: int,
    irt_scores: Optional[np.ndarray] = None,
    criterion: str = 'irt',
) -> List[int]:
    """
    Select k representative items from a feature matrix using spectral
    clustering, then picking one representative per cluster.

    Args:
        X:          Feature matrix, shape (n_items, n_features).
        k:          Number of items to select (= number of clusters).
        irt_scores: Per-item IRT info scores, shape (n_items,).
                    Required when criterion='irt'.
        criterion:  How to pick the representative from each cluster.
                    'irt'        → highest IRT score  (medium stratum)
                    'centrality' → closest to centroid (easy/hard strata)

    Returns:
        List of local indices (into X) of selected items.

    Fallback behaviour:
        If n < 2k or clustering raises an exception, falls back to
        _fallback_select() which uses IRT top-k or even spacing.
    """
    n = len(X)
    k = min(k, n)
    if k <= 0:
        return []
    if k == n:
        return list(range(n))

    # Degenerate: too few items for clustering to be meaningful
    if n < 2 * k:
        return _fallback_select(n, k, irt_scores, criterion)

    X_norm = normalize(X, norm='l2') if X.ndim == 2 and X.shape[1] > 1 else X

    S = build_similarity_matrix(X_norm)

    try:
        clust = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=10,
        )
        labels = clust.fit_predict(S)
    except Exception as exc:
        logger.warning(f'SpectralClustering failed ({exc}); falling back to top-k.')
        return _fallback_select(n, k, irt_scores, criterion)

    selected: List[int] = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            # Rare: fewer non-empty clusters than k (can happen with duplicates)
            continue

        if criterion == 'irt' and irt_scores is not None:
            rep = int(members[np.argmax(irt_scores[members])])
        else:
            # Centrality: item closest to cluster mean in normalised embedding space
            centroid = X_norm[members].mean(axis=0)
            dists = np.linalg.norm(X_norm[members] - centroid, axis=1)
            rep = int(members[np.argmin(dists)])

        selected.append(rep)

    # If some clusters were empty, top-up with highest-IRT un-selected items
    if len(selected) < k:
        selected_set = set(selected)
        remaining = [i for i in range(n) if i not in selected_set]
        if irt_scores is not None:
            remaining.sort(key=lambda i: irt_scores[i], reverse=True)
        selected.extend(remaining[:k - len(selected)])

    return selected


def _fallback_select(
    n: int,
    k: int,
    irt_scores: Optional[np.ndarray],
    criterion: str,
) -> List[int]:
    """
    Select k items without clustering (used when n < 2k).

    - criterion='irt':        top-k by IRT info score
    - criterion='centrality': evenly-spaced by index (maximises spread)
    """
    if criterion == 'irt' and irt_scores is not None:
        order = np.argsort(irt_scores)[::-1]
        return [int(i) for i in order[:k]]
    else:
        # Evenly-spaced indices ensure coverage of the stratum
        indices = np.round(np.linspace(0, n - 1, k)).astype(int)
        return [int(i) for i in indices]
