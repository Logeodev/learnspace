# How to guess an initial MLP topology

## 1) What we try to do

We describe a *heuristic* (rule‑of‑thumb) for **guessing an initial MLP topology** (number of hidden layers + number of neurons per layer), using unsupervised structure in your data:

### Step A — Dimensionality reduction (PCA or Kernel PCA)

*   You apply **PCA** (linear) or **KernelPCA** (nonlinear) to compress your features into fewer latent dimensions.
*   You keep enough components to explain a target fraction of variance (they propose \~40% for binary, \~60–70% for multiclass ; instead of the usual \~95-99%).  
    ✅ This is **not a law**—just a heuristic to avoid keeping too many components.

**Interpretation:** the number of retained components `n_components` becomes an upper bound on model complexity.

### Step B — Convert “components” into a network depth guess

The guess here is that the number of components can inform how many maximum “stages” of representation the MLP might need.

A reasonable interpretation is:

*   Let **`n_components`** (given in A) ≈ a reasonable maximum number of “stages” of representation.
*   Pick a **small maximum depth** (e.g., 1–3 layers) even if `n_components` is large.

### Step C — Use clustering to estimate neurons per layer

*   Project data into PC space (`X_pca_scores`)
*   Run **KMeans** with different `k` (number of clusters).
*   Use **Elbow** or **Silhouette** to pick a good `k`.
*   Map that `k` to **neurons** for that hidden layer.

A practical variant:

*   For layer 1: cluster using first `d1` PCs → choose `k1` → neurons = `k1`
*   For layer 2: cluster using first `d2 > d1` PCs → choose `k2` → neurons = `k2`
*   etc.

### Step D — Validate topology stability under regularization

*   Train MLPs with the proposed `hidden_layer_sizes`.
*   Vary regularization strength `alpha`.
*   If topology is “right-ish”, performance should not collapse when `alpha` changes.

> ⚠️ This is **only a starting heuristic**. Final tuning is typically done with grid search / Bayesian optimization / random search.

***

## 2) Python / scikit‑learn script implementing the heuristic

### What this script does

1.  Chooses PCA or KernelPCA.
2.  Chooses number of components by target “variance”:
    *   For PCA: true explained variance.
    *   For KernelPCA: approximate via normalized eigenvalues (in feature space).
3.  Estimates hidden layer sizes via KMeans:
    *   Picks `k` using **silhouette** (primary) + elbow fallback.
4.  Trains and evaluates an `MLPClassifier` across multiple `alpha` values.
5.  Reports mean CV accuracy and stability.

> Works for classification. (You can adapt to regression by using `MLPRegressor` and R²/MAE.)

***

### Implementation

```python
import numpy as np

from sklearn.base import clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------
# Helpers: choose components
# -------------------------
def choose_pca_components(X, variance_target=0.65, random_state=0):
    """
    PCA with n_components as a float keeps enough components to reach variance_target.
    Returns fitted PCA and transformed X.
    """
    pca = PCA(n_components=variance_target, svd_solver="full", random_state=random_state)
    X_red = pca.fit_transform(X)
    return pca, X_red


def choose_kpca_components(X, variance_target=0.65, kernel="rbf", gamma=None,
                          n_max=50, random_state=0):
    """
    KernelPCA doesn't provide explained_variance_ratio_ like PCA.
    We'll fit with many components, then approximate "variance explained" by
    normalized eigenvalues_ in feature space.

    Returns fitted KPCA with chosen n_components and transformed X.
    """
    n_max = int(min(n_max, X.shape[1], X.shape[0] - 1))
    kpca_full = KernelPCA(
        n_components=n_max,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=False,
        random_state=random_state,
        eigen_solver="auto"
    )
    X_full = kpca_full.fit_transform(X)

    # Approximate "variance explained" in feature space using eigenvalues_
    eig = np.array(kpca_full.eigenvalues_, dtype=float)
    eig = np.maximum(eig, 0)  # numerical safety
    ratio = eig / (eig.sum() + 1e-12)
    cum = np.cumsum(ratio)

    n_keep = int(np.searchsorted(cum, variance_target) + 1)
    n_keep = max(1, min(n_keep, n_max))

    # Refit KPCA with chosen components for cleanliness
    kpca = KernelPCA(
        n_components=n_keep,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=False,
        random_state=random_state,
        eigen_solver="auto"
    )
    X_red = kpca.fit_transform(X)
    return kpca, X_red


# -------------------------
# Helpers: choose K for KMeans
# -------------------------
def elbow_k(inertias, k_values):
    """
    Simple elbow heuristic using second difference.
    Not perfect, but decent fallback when silhouette is not usable.
    """
    inertias = np.array(inertias, dtype=float)
    if len(inertias) < 3:
        return k_values[np.argmin(inertias)]
    # second derivative approximation
    d1 = np.diff(inertias)
    d2 = np.diff(d1)
    elbow_idx = np.argmax(-d2) + 2  # shift due to diffs
    elbow_idx = np.clip(elbow_idx, 0, len(k_values) - 1)
    return k_values[elbow_idx]


def choose_k_by_silhouette_or_elbow(X, k_min=2, k_max=20, random_state=0):
    """
    Picks k by maximizing silhouette score when possible.
    Falls back to elbow on inertia.
    """
    k_max = min(k_max, X.shape[0] - 1)  # silhouette needs at least 2 clusters
    if k_max < k_min:
        return 1  # degenerate case

    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)

        # silhouette requires >1 cluster and < n_samples clusters
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan
        silhouettes.append(sil)

    silhouettes = np.array(silhouettes, dtype=float)

    # Prefer silhouette if any finite values exist
    if np.isfinite(silhouettes).any():
        best_k = k_values[int(np.nanargmax(silhouettes))]
        return best_k

    # Fallback to elbow
    return elbow_k(inertias, k_values)


# -------------------------
# Main heuristic: estimate MLP sizes
# -------------------------
def estimate_mlp_hidden_layers(
    X, y,
    method="pca",                  # "pca" or "kpca"
    variance_target=None,          # if None: binary->0.40 else ->0.65
    max_layers=3,                  # keep depth small for practicality
    pcs_per_layer="increasing",    # "increasing" or "all"
    k_range=(2, 20),
    kpca_kernel="rbf",
    kpca_gamma=None,
    random_state=0
):
    """
    Returns:
      hidden_layer_sizes (tuple),
      reducer (fitted PCA/KPCA),
      X_red (reduced representation)
    """
    # Decide default variance target based on number of classes
    n_classes = len(np.unique(y))
    if variance_target is None:
        variance_target = 0.40 if n_classes == 2 else 0.65

    # Scale then reduce
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if method.lower() == "pca":
        reducer, X_red = choose_pca_components(Xs, variance_target, random_state=random_state)
        n_comp = X_red.shape[1]
    elif method.lower() == "kpca":
        reducer, X_red = choose_kpca_components(
            Xs, variance_target,
            kernel=kpca_kernel, gamma=kpca_gamma,
            n_max=50, random_state=random_state
        )
        n_comp = X_red.shape[1]
    else:
        raise ValueError("method must be 'pca' or 'kpca'")

    # Depth heuristic: cap layers
    n_layers = int(min(max_layers, max(1, n_comp)))

    # Estimate neurons per layer via KMeans on PC scores
    neurons = []
    for layer_idx in range(1, n_layers + 1):
        if pcs_per_layer == "increasing":
            d = min(layer_idx, n_comp)  # 1 PC for layer 1, 2 PCs for layer 2, ...
        else:
            d = n_comp  # use all PCs for every layer

        X_layer = X_red[:, :d]
        k = choose_k_by_silhouette_or_elbow(
            X_layer, k_min=k_range[0], k_max=k_range[1], random_state=random_state
        )
        neurons.append(int(k))

    hidden_layer_sizes = tuple(neurons)
    return hidden_layer_sizes, (scaler, reducer), X_red


# -------------------------
# Evaluate topology stability across alpha
# -------------------------
def evaluate_topology_with_alphas(X, y, hidden_layer_sizes, alphas=(1e-5, 1e-4, 1e-3, 1e-2),
                                  random_state=0, cv_splits=5):
    """
    CV accuracy across multiple regularization strengths (alpha).
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results = {}
    for alpha in alphas:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=500,
            random_state=random_state
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", clf)
        ])

        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        results[alpha] = (scores.mean(), scores.std())
    return results


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris

    # --- Binary example
    data = load_breast_cancer()
    X, y = data.data, data.target

    hidden, reducer, X_red = estimate_mlp_hidden_layers(
        X, y,
        method="pca",
        variance_target=None,   # auto: binary -> 0.40
        max_layers=3,
        pcs_per_layer="increasing",
        k_range=(2, 25),
        random_state=0
    )

    print("Binary dataset (breast cancer)")
    print("Estimated hidden_layer_sizes:", hidden)
    res = evaluate_topology_with_alphas(X, y, hidden)
    for a, (m, s) in res.items():
        print(f"alpha={a:>7}: accuracy={m:.4f} ± {s:.4f}")

    print("\n" + "-"*60 + "\n")

    # --- Multiclass example
    data = load_iris()
    X, y = data.data, data.target

    hidden, reducer, X_red = estimate_mlp_hidden_layers(
        X, y,
        method="kpca",
        variance_target=None,   # auto: multiclass -> 0.65
        max_layers=3,
        pcs_per_layer="increasing",
        k_range=(2, 10),
        kpca_kernel="rbf",
        kpca_gamma=1.0,
        random_state=0
    )

    print("Multiclass dataset (iris)")
    print("Estimated hidden_layer_sizes:", hidden)
    res = evaluate_topology_with_alphas(X, y, hidden)
    for a, (m, s) in res.items():
        print(f"alpha={a:>7}: accuracy={m:.4f} ± {s:.4f}")
```

***

## 3) Notes & caveats

### PCA variance thresholds (40% vs 60–70%)

*   These thresholds are **not standard**. In many real problems, people keep **80–99%** variance in PCA when using PCA as a lossless-ish compression step.
*   Here, the idea is different: **use PCA as a complexity prior**, not as faithful reconstruction.

### KernelPCA “variance explained”

*   KernelPCA doesn’t have the same direct explained variance concept as PCA.
*   Using normalized eigenvalues is a **reasonable approximation in feature space**, but treat it as a heuristic.

### Clusters ≠ neurons

KMeans clusters are not “the number of neurons you need”.  
But as a **starting guess**, “how many groups exist in the latent space” can correlate with “how much capacity the network might need”.

### Topology stability under alpha

Good idea: if a topology only works at one narrow alpha, it might be brittle.  
But also: changing alpha can change the “best” topology—so interpret “doesn’t change much” as *a sign of robustness*, not a guarantee.

