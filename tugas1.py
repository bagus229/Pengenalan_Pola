import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. Implementasi KNN dari Nol
# ─────────────────────────────────────────
class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


# ─────────────────────────────────────────
# 2. Load & Preprocessing Dataset
# ─────────────────────────────────────────
print("=" * 60)
print("TUGAS 1: IMPLEMENTASI KNN DARI NOL")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target
print(f"\nDataset   : Iris")
print(f"Sampel    : {X.shape[0]}")
print(f"Fitur     : {X.shape[1]}")
print(f"Kelas     : {np.unique(y)} -> {iris.target_names.tolist()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 3. 5-Fold Cross-Validation untuk K = [1,3,5,7,10,15]
# ─────────────────────────────────────────
K_values = [1, 3, 5, 7, 10, 15]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_scratch  = {k: [] for k in K_values}
results_sklearn  = {k: [] for k in K_values}

print("\nMemulai 5-Fold Cross-Validation ...")
for k in K_values:
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_scaled, y)):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # KNN Scratch
        knn_scratch = KNNFromScratch(k=k)
        knn_scratch.fit(X_tr, y_tr)
        y_pred_scratch = knn_scratch.predict(X_te)
        acc_scratch = accuracy_score(y_te, y_pred_scratch)
        results_scratch[k].append(acc_scratch)

        # KNN Sklearn
        knn_sk = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn_sk.fit(X_tr, y_tr)
        y_pred_sk = knn_sk.predict(X_te)
        acc_sk = accuracy_score(y_te, y_pred_sk)
        results_sklearn[k].append(acc_sk)

# ─────────────────────────────────────────
# 4. Tampilkan Hasil
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'K':>4} | {'KNN Scratch (mean±std)':^25} | {'KNN Sklearn (mean±std)':^25} | {'Selisih':^10}")
print("-" * 72)

mean_scratch_list, mean_sklearn_list = [], []
for k in K_values:
    ms = np.mean(results_scratch[k])
    ss = np.std(results_scratch[k])
    mk = np.mean(results_sklearn[k])
    sk = np.std(results_sklearn[k])
    diff = abs(ms - mk)
    mean_scratch_list.append(ms)
    mean_sklearn_list.append(mk)
    print(f"{k:>4} | {ms:.4f} ± {ss:.4f}          | {mk:.4f} ± {sk:.4f}          | {diff:.6f}")

best_k_scratch = K_values[np.argmax(mean_scratch_list)]
best_k_sklearn = K_values[np.argmax(mean_sklearn_list)]
print("-" * 72)
print(f"\nBest K (Scratch) : K={best_k_scratch}  -> Akurasi={max(mean_scratch_list):.4f}")
print(f"Best K (Sklearn) : K={best_k_sklearn}  -> Akurasi={max(mean_sklearn_list):.4f}")

# ─────────────────────────────────────────
# 5. Visualisasi
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Tugas 1: KNN dari Nol vs Sklearn - Dataset Iris", fontsize=14, fontweight='bold')

# Plot 1: Akurasi vs K
axes[0].plot(K_values, mean_scratch_list, 'bo-', linewidth=2, markersize=8, label='KNN Scratch')
axes[0].plot(K_values, mean_sklearn_list, 'rs--', linewidth=2, markersize=8, label='KNN Sklearn')
axes[0].set_xlabel("Nilai K", fontsize=12)
axes[0].set_ylabel("Akurasi Rata-rata (5-Fold CV)", fontsize=12)
axes[0].set_title("Akurasi vs Nilai K", fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(K_values)
axes[0].set_ylim(0.9, 1.01)

# Plot 2: Selisih akurasi Scratch vs Sklearn
diffs = [abs(mean_scratch_list[i] - mean_sklearn_list[i]) for i in range(len(K_values))]
bars = axes[1].bar(K_values, [d * 100 for d in diffs], color='steelblue', alpha=0.7, width=1.5)
axes[1].set_xlabel("Nilai K", fontsize=12)
axes[1].set_ylabel("Selisih Akurasi (%)", fontsize=12)
axes[1].set_title("Selisih Akurasi: Scratch vs Sklearn", fontsize=12)
axes[1].set_xticks(K_values)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, diffs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{d*100:.4f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/tugas1_hasil.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik disimpan: tugas1_hasil.png")

# ─────────────────────────────────────────
# 6. Analisis
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALISIS:")
print("=" * 60)
print(f"- Implementasi KNN dari nol menggunakan jarak Euclidean.")
print(f"- Selisih akurasi antara KNN Scratch dan Sklearn sangat kecil")
print(f"  (mendekati 0), membuktikan implementasi scratch sudah benar.")
print(f"- K terlalu kecil (K=1) cenderung overfit (sensitif noise).")
print(f"- K terlalu besar bisa underfit karena meratakan keputusan.")
print(f"- Best K pada dataset Iris: K={best_k_scratch} (akurasi={max(mean_scratch_list):.4f})")