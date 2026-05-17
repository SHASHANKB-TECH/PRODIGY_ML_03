import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFilter

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE       = 64          # resize target (px)
N_SAMPLES      = 1000        # synthetic samples per class (ignored if real data found)
PCA_COMPONENTS = 80          # dimensions after PCA
RANDOM_STATE   = 42

KAGGLE_DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", "")   # e.g. /root/.cache/kagglehub/...

# ── Palette ───────────────────────────────────────────────────────────────────
BG, PANEL, BORDER = "#0a0f1e", "#111827", "#1f2d45"
TEXT, MUTED       = "#e2e8f0", "#64748b"
CAT_C, DOG_C      = "#f472b6", "#38bdf8"
GREEN, ORANGE     = "#4ade80", "#fb923c"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING / GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def hog_features(img_gray: np.ndarray, cell=8, block=2, bins=9) -> np.ndarray:
    """Lightweight HOG without skimage dependency."""
    # Gradients
    gx = np.zeros_like(img_gray, dtype=np.float32)
    gy = np.zeros_like(img_gray, dtype=np.float32)
    gx[:, 1:-1] = img_gray[:, 2:].astype(np.float32) - img_gray[:, :-2].astype(np.float32)
    gy[1:-1, :] = img_gray[2:, :].astype(np.float32) - img_gray[:-2, :].astype(np.float32)
    mag   = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180

    h, w = img_gray.shape
    n_cells_y, n_cells_x = h // cell, w // cell
    hist = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)

    bin_width = 180 / bins
    for b in range(bins):
        lo, hi = b * bin_width, (b + 1) * bin_width
        mask = (angle >= lo) & (angle < hi)
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                cell_mask = mask[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
                cell_mag  = mag [cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
                hist[cy, cx, b] = cell_mag[cell_mask].sum()

    # Block normalisation
    features = []
    for by in range(n_cells_y - block + 1):
        for bx in range(n_cells_x - block + 1):
            block_hist = hist[by:by+block, bx:bx+block, :].ravel()
            norm = np.linalg.norm(block_hist) + 1e-6
            features.append(block_hist / norm)

    return np.concatenate(features)


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return hog_features(np.array(img))


def generate_synthetic_animal(label: int, rng: np.random.Generator) -> np.ndarray:
    """
    Procedurally generate a plausible grayscale animal silhouette.
    Cats  (label=0): rounder head, pointed ears, lighter coat texture
    Dogs  (label=1): longer snout blob, floppy ear blobs, darker coat texture
    """
    arr = np.full((IMG_SIZE, IMG_SIZE), 180, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

    if label == 0:  # cat
        # head — round
        r = int(rng.integers(18, 24))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=int(rng.integers(80, 130)))
        # pointed ears
        for sign in (-1, 1):
            ex = cx + sign * int(r * 0.65)
            draw.polygon([(ex-6, cy-r+4), (ex+6, cy-r+4), (ex, cy-r-12)],
                         fill=int(rng.integers(60, 110)))
        # eyes
        for sign in (-1, 1):
            draw.ellipse([cx+sign*8-3, cy-5-3, cx+sign*8+3, cy-5+3],
                         fill=int(rng.integers(200, 240)))
        # nose
        draw.ellipse([cx-2, cy+2, cx+2, cy+6], fill=int(rng.integers(190, 220)))
    else:  # dog
        # head — slightly oval
        draw.ellipse([cx-20, cy-18, cx+20, cy+22], fill=int(rng.integers(90, 145)))
        # floppy ears
        for sign in (-1, 1):
            ex0, ex1 = sorted([cx+sign*14, cx+sign*28])
            draw.ellipse([ex0, cy-10, ex1, cy+14],
                         fill=int(rng.integers(60, 100)))
        # snout blob
        draw.ellipse([cx-8, cy+4, cx+8, cy+16], fill=int(rng.integers(110, 155)))
        # eyes
        for sign in (-1, 1):
            draw.ellipse([cx+sign*8-4, cy-8-4, cx+sign*8+4, cy-8+4],
                         fill=int(rng.integers(30, 70)))

    # Add noise texture
    img_arr = np.array(img, dtype=np.float32)
    img_arr += rng.normal(0, 12, img_arr.shape)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr).filter(ImageFilter.GaussianBlur(radius=0.8))
    return hog_features(np.array(img))


def load_kaggle_images(data_dir: str, max_per_class=1000):
    files = os.listdir(data_dir)
    cat_files = [f for f in files if f.startswith("cat.")][:max_per_class]
    dog_files = [f for f in files if f.startswith("dog.")][:max_per_class]
    if not cat_files or not dog_files:
        return None, None
    X, y = [], []
    for f in cat_files:
        X.append(load_image(os.path.join(data_dir, f))); y.append(0)
    for f in dog_files:
        X.append(load_image(os.path.join(data_dir, f))); y.append(1)
    return np.array(X), np.array(y)


print("=" * 60)
print("  SVM Cats vs Dogs Classifier")
print("=" * 60)

# Try real data first
X, y, data_source = None, None, "synthetic"
if KAGGLE_DATA_DIR and os.path.isdir(KAGGLE_DATA_DIR):
    print(f"\n  Loading Kaggle images from: {KAGGLE_DATA_DIR}")
    X, y = load_kaggle_images(KAGGLE_DATA_DIR)
    if X is not None:
        data_source = "kaggle"
        print(f"  Loaded {len(X)} images (HOG features: {X.shape[1]})")

if X is None:
    print(f"\n  Kaggle data unavailable — generating {N_SAMPLES*2} synthetic animal images")
    print("  (Set KAGGLE_DATA_DIR env var to use real images)\n")
    rng = np.random.default_rng(RANDOM_STATE)
    X = np.array([generate_synthetic_animal(lbl, rng)
                  for lbl in [0]*N_SAMPLES + [1]*N_SAMPLES])
    y = np.array([0]*N_SAMPLES + [1]*N_SAMPLES)
    # Inject realistic noise correlations
    X += rng.normal(0, 0.03, X.shape)
    print(f"  HOG feature vector length: {X.shape[1]}")

print(f"  Data source : {data_source}")
print(f"  Total samples: {len(X)}  (cats={int((y==0).sum())}, dogs={int((y==1).sum())})")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PIPELINE: Scaler → PCA → SVM
# ══════════════════════════════════════════════════════════════════════════════
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE, whiten=True)),
    ("svm",    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
])

param_grid = {
    "svm__C":     [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001],
}

print("\n  Running GridSearchCV (4×3 grid, 5-fold CV) ...")
t0 = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy",
                    n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)
elapsed = time.time() - t0

best_params = grid.best_params_
best_cv_score = grid.best_score_
print(f"  Done in {elapsed:.1f}s")
print(f"  Best params : {best_params}")
print(f"  Best CV acc : {best_cv_score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
best_model = grid.best_estimator_
y_pred     = best_model.predict(X_test)
y_prob     = best_model.predict_proba(X_test)[:, 1]

acc   = accuracy_score(y_test, y_pred)
cm    = confusion_matrix(y_test, y_pred)
report= classification_report(y_test, y_pred, target_names=["Cat", "Dog"])
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# PCA variance explained
pca_fitted = best_model.named_steps["pca"]
var_exp = np.cumsum(pca_fitted.explained_variance_ratio_)

# GridSearchCV results table
cv_results = grid.cv_results_

print(f"\n  Test Accuracy : {acc:.4f}")
print(f"  ROC AUC      : {roc_auc:.4f}")
print(f"\n{report}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.44, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

def style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

# ── 5a. Confusion matrix ─────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[0, 0])
cmap = plt.cm.colors.LinearSegmentedColormap.from_list("cm", [PANEL, CAT_C])
im = ax_cm.imshow(cm, cmap=cmap, aspect="auto")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                   color=TEXT, fontsize=16, fontweight="bold")
ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Cat", "Dog"], color=TEXT, fontsize=9)
ax_cm.set_yticklabels(["Cat", "Dog"], color=TEXT, fontsize=9, rotation=90, va="center")
style(ax_cm, "Confusion Matrix", "Predicted", "Actual")

# ── 5b. ROC curve ────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, 1])
ax_roc.plot(fpr, tpr, color=DOG_C, lw=2, label=f"AUC = {roc_auc:.3f}")
ax_roc.fill_between(fpr, tpr, alpha=0.12, color=DOG_C)
ax_roc.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--")
ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
style(ax_roc, "ROC Curve", "False Positive Rate", "True Positive Rate")
ax_roc.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# ── 5c. Metrics card ─────────────────────────────────────────────────────────
ax_card = fig.add_subplot(gs[0, 2])
ax_card.set_facecolor(PANEL)
ax_card.axis("off")
style(ax_card, "Model Performance")

lines = report.strip().split("\n")
cat_line = [l for l in lines if "Cat" in l][0].split()
dog_line = [l for l in lines if "Dog" in l][0].split()

metrics = [
    ("Accuracy",       f"{acc:.4f}"),
    ("ROC AUC",        f"{roc_auc:.4f}"),
    ("CV Accuracy",    f"{best_cv_score:.4f}"),
    ("",               ""),
    ("Cat Precision",  cat_line[1]),
    ("Cat Recall",     cat_line[2]),
    ("Cat F1",         cat_line[3]),
    ("",               ""),
    ("Dog Precision",  dog_line[1]),
    ("Dog Recall",     dog_line[2]),
    ("Dog F1",         dog_line[3]),
    ("",               ""),
    ("Best C",         str(best_params["svm__C"])),
    ("Best γ",         str(best_params["svm__gamma"])),
]
for i, (label, val) in enumerate(metrics):
    if not label:
        continue
    yp = 0.95 - i * 0.065
    col = CAT_C if "Cat" in label else (DOG_C if "Dog" in label else GREEN)
    ax_card.text(0.05, yp, label, transform=ax_card.transAxes, color=MUTED, fontsize=8)
    ax_card.text(0.95, yp, val,   transform=ax_card.transAxes, color=col,
                 fontsize=8, fontweight="bold", ha="right")

# ── 5d. PCA explained variance ───────────────────────────────────────────────
ax_pca = fig.add_subplot(gs[1, 0])
components = np.arange(1, len(var_exp) + 1)
ax_pca.plot(components, var_exp * 100, color=GREEN, lw=2)
ax_pca.fill_between(components, var_exp * 100, alpha=0.12, color=GREEN)
ax_pca.axhline(90, color=ORANGE, lw=1, linestyle="--", alpha=0.7)
ax_pca.text(PCA_COMPONENTS * 0.6, 91, "90% threshold", color=ORANGE, fontsize=7)
style(ax_pca, f"PCA Cumulative Variance ({PCA_COMPONENTS} components)",
      "Components", "Variance Explained (%)")
ax_pca.set_ylim(0, 102)

# ── 5e. Prediction probability distribution ───────────────────────────────────
ax_dist = fig.add_subplot(gs[1, 1])
ax_dist.hist(y_prob[y_test == 0], bins=25, color=CAT_C, alpha=0.65,
             label="True Cat", edgecolor=BG, lw=0.4)
ax_dist.hist(y_prob[y_test == 1], bins=25, color=DOG_C, alpha=0.65,
             label="True Dog", edgecolor=BG, lw=0.4)
ax_dist.axvline(0.5, color="white", lw=1, linestyle="--")
style(ax_dist, "P(Dog) Score Distribution", "Predicted P(Dog)", "Count")
ax_dist.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# ── 5f. GridSearch heatmap ────────────────────────────────────────────────────
ax_gs = fig.add_subplot(gs[1, 2])
C_vals     = sorted(set(p["svm__C"] for p in grid.cv_results_["params"]))
gamma_vals = sorted(set(str(p["svm__gamma"]) for p in grid.cv_results_["params"]),
                    key=lambda g: (0, 0) if g == "scale" else (1, float(g)))
score_grid = np.zeros((len(gamma_vals), len(C_vals)))
for p, s in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"]):
    ci = C_vals.index(p["svm__C"])
    gi = gamma_vals.index(str(p["svm__gamma"]))
    score_grid[gi, ci] = s

gs_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("gs", [PANEL, DOG_C])
ax_gs.imshow(score_grid, cmap=gs_cmap, aspect="auto", vmin=score_grid.min())
ax_gs.set_xticks(range(len(C_vals))); ax_gs.set_xticklabels(C_vals, color=TEXT, fontsize=8)
ax_gs.set_yticks(range(len(gamma_vals))); ax_gs.set_yticklabels(gamma_vals, color=TEXT, fontsize=8)
for gi in range(len(gamma_vals)):
    for ci in range(len(C_vals)):
        ax_gs.text(ci, gi, f"{score_grid[gi, ci]:.3f}", ha="center", va="center",
                   color=TEXT, fontsize=7.5)
style(ax_gs, "GridSearchCV Accuracy Heatmap", "C", "gamma")

# ── 5g. Per-class bar comparison ─────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, 0])
cats = ["Precision", "Recall", "F1-Score"]
cat_v = [float(cat_line[1]), float(cat_line[2]), float(cat_line[3])]
dog_v = [float(dog_line[1]), float(dog_line[2]), float(dog_line[3])]
x = np.arange(len(cats))
w = 0.35
ax_bar.bar(x - w/2, cat_v, w, color=CAT_C, alpha=0.85, label="Cat", edgecolor=BG)
ax_bar.bar(x + w/2, dog_v, w, color=DOG_C, alpha=0.85, label="Dog", edgecolor=BG)
ax_bar.set_xticks(x); ax_bar.set_xticklabels(cats)
ax_bar.set_ylim(0, 1.1)
for xi, (cv, dv) in enumerate(zip(cat_v, dog_v)):
    ax_bar.text(xi - w/2, cv + 0.02, f"{cv:.2f}", ha="center", color=CAT_C, fontsize=7.5, fontweight="bold")
    ax_bar.text(xi + w/2, dv + 0.02, f"{dv:.2f}", ha="center", color=DOG_C, fontsize=7.5, fontweight="bold")
style(ax_bar, "Per-Class Metrics", "", "Score")
ax_bar.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# ── 5h. PCA component scatter (first 2 PCs) ──────────────────────────────────
ax_pca2 = fig.add_subplot(gs[2, 1])
pipe_pca = Pipeline([("sc", best_model.named_steps["scaler"]),
                     ("pc", best_model.named_steps["pca"])])
X_test_pca = pipe_pca.transform(X_test)
for lbl, color, name in [(0, CAT_C, "Cat"), (1, DOG_C, "Dog")]:
    mask = y_test == lbl
    ax_pca2.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                    color=color, alpha=0.5, s=18, edgecolors="none", label=name)
style(ax_pca2, "PCA Space — Test Set (PC1 vs PC2)", "PC 1", "PC 2")
ax_pca2.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# ── 5i. Decision boundary (PC1 vs PC2) ───────────────────────────────────────
ax_db = fig.add_subplot(gs[2, 2])
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Build a lightweight SVM on 2 PCs for visualisation
svm_2d = SVC(kernel="rbf", C=best_params["svm__C"],
             gamma=best_params["svm__gamma"], random_state=RANDOM_STATE)
svm_2d.fit(X_test_pca[:, :2], y_test)
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

db_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
    "db", [CAT_C + "30", DOG_C + "30"])
ax_db.contourf(xx, yy, Z, alpha=0.35, cmap=db_cmap)
ax_db.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=0.8)
for lbl, color in [(0, CAT_C), (1, DOG_C)]:
    mask = y_test == lbl
    ax_db.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                  color=color, alpha=0.6, s=16, edgecolors="none")
style(ax_db, "Decision Boundary (2-D PCA)", "PC 1", "PC 2")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"SVM Cats vs Dogs  ·  HOG Features  ·  PCA({PCA_COMPONENTS})  ·  "
    f"RBF Kernel  ·  Test Accuracy = {acc:.3f}  ·  AUC = {roc_auc:.3f}",
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)
out = "svm_cats_dogs.png"

plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

print(f"\nDashboard saved → {out}")
