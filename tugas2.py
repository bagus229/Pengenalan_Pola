import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# PATH DATASET
# ============================================================

DATASET_PATH = "dataset_texture"

# ============================================================
# LOAD DATASET
# Struktur:
#
# dataset/
#    class1/
#    class2/
#    class3/
# ============================================================

images = []
labels = []

classes = os.listdir(DATASET_PATH)

print("CLASS:")
print(classes)

for label in classes:

    class_path = os.path.join(DATASET_PATH, label)

    for filename in os.listdir(class_path):

        img_path = os.path.join(class_path, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (128, 128))

        images.append(img)

        labels.append(label)

images = np.array(images)
labels = np.array(labels)

print("\nJumlah gambar :", len(images))

# ============================================================
# FEATURE EXTRACTION
# ============================================================

# ----------------------------
# LBP
# ----------------------------

def extract_lbp(image):

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(
        image,
        n_points,
        radius,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-6)

    return hist

# ----------------------------
# HOG
# ----------------------------

def extract_hog(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    return features

# ----------------------------
# GLCM
# ----------------------------

def extract_glcm(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return np.array([
        contrast,
        correlation,
        energy,
        homogeneity
    ])

# ============================================================
# EKSTRAKSI FITUR
# ============================================================

print("\nEkstraksi fitur...")

X_lbp = np.array([extract_lbp(img) for img in images])

X_hog = np.array([extract_hog(img) for img in images])

X_glcm = np.array([extract_glcm(img) for img in images])

# ============================================================
# SPLIT DATA
# ============================================================

X_train_lbp, X_test_lbp, y_train, y_test = train_test_split(
    X_lbp,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

X_train_hog, X_test_hog, _, _ = train_test_split(
    X_hog,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

X_train_glcm, X_test_glcm, _, _ = train_test_split(
    X_glcm,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ============================================================
# FUNCTION EVALUASI
# ============================================================

def evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test,
    classifier,
    feature_name,
    classifier_name
):

    print("\n" + "="*60)
    print(f"{feature_name} + {classifier_name}")
    print("="*60)

    # TRAINING
    classifier.fit(X_train, y_train)

    # PREDIKSI
    y_pred = classifier.predict(X_test)

    # ACCURACY
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f}")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.unique(labels)
    )

    disp.plot(cmap='Blues')

    plt.title(f"Confusion Matrix\n{feature_name} + {classifier_name}")

    plt.show()

    return acc

# ============================================================
# MODEL
# ============================================================

svm = SVC()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

results = []

# ============================================================
# LBP
# ============================================================

acc_lbp_svm = evaluate_model(
    X_train_lbp,
    X_test_lbp,
    y_train,
    y_test,
    svm,
    "LBP",
    "SVM"
)

results.append(["LBP", "SVM", acc_lbp_svm])

acc_lbp_rf = evaluate_model(
    X_train_lbp,
    X_test_lbp,
    y_train,
    y_test,
    rf,
    "LBP",
    "Random Forest"
)

results.append(["LBP", "Random Forest", acc_lbp_rf])

# ============================================================
# HOG
# ============================================================

acc_hog_svm = evaluate_model(
    X_train_hog,
    X_test_hog,
    y_train,
    y_test,
    svm,
    "HOG",
    "SVM"
)

results.append(["HOG", "SVM", acc_hog_svm])

acc_hog_rf = evaluate_model(
    X_train_hog,
    X_test_hog,
    y_train,
    y_test,
    rf,
    "HOG",
    "Random Forest"
)

results.append(["HOG", "Random Forest", acc_hog_rf])

# ============================================================
# GLCM
# ============================================================

acc_glcm_svm = evaluate_model(
    X_train_glcm,
    X_test_glcm,
    y_train,
    y_test,
    svm,
    "GLCM",
    "SVM"
)

results.append(["GLCM", "SVM", acc_glcm_svm])

acc_glcm_rf = evaluate_model(
    X_train_glcm,
    X_test_glcm,
    y_train,
    y_test,
    rf,
    "GLCM",
    "Random Forest"
)

results.append(["GLCM", "Random Forest", acc_glcm_rf])

# ============================================================
# HASIL AKHIR
# ============================================================

df = pd.DataFrame(results, columns=[
    "Feature",
    "Classifier",
    "Accuracy"
])

print("\n")
print("="*60)
print("HASIL AKHIR")
print("="*60)

print(df)

# ============================================================
# VISUALISASI AKURASI
# ============================================================

labels_plot = [
    "LBP-SVM",
    "LBP-RF",
    "HOG-SVM",
    "HOG-RF",
    "GLCM-SVM",
    "GLCM-RF"
]

accuracies = df["Accuracy"]

plt.figure(figsize=(12, 5))

bars = plt.bar(labels_plot, accuracies)

plt.title("Perbandingan Akurasi Fitur dan Classifier")

plt.xlabel("Metode")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

# tampilkan nilai accuracy di atas bar
for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.01,
        round(yval, 3),
        ha='center'
    )

plt.show()

