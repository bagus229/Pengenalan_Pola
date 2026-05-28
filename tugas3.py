import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
import time
import json

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint, CSVLogger)
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

CONFIG = {
    'img_size'    : 128,          # ukuran input gambar (128x128)
    'batch_size'  : 32,
    'epochs'      : 30,
    'learning_rate': 1e-3,
    'fine_tune_lr' : 1e-5,
    'fine_tune_at' : 10,          # epoch mulai fine-tuning
    'dataset_dir' : './dataset',  # ← GANTI dengan path dataset Anda
    'seed'        : 42,
    'dropout'     : 0.4,
}

tf.random.set_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# ─────────────────────────────────────────────
# 1. Persiapan Dataset
# ─────────────────────────────────────────────

def prepare_data_generators(dataset_dir, img_size, batch_size):
    """
    Buat data generator dengan augmentasi untuk training.
    Mendukung struktur: dataset/train/<kelas>/ dan dataset/test/<kelas>/
    """
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir  = os.path.join(dataset_dir, 'test')

    print(f"Train dir: {train_dir}")
    print(f"Test dir : {test_dir}")

    # Augmentasi untuk training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.15,          # 15% untuk validasi
    )

    # Hanya rescale untuk test
    test_datagen = ImageDataGenerator(rescale=1./255)

    target = (img_size, img_size)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=target, batch_size=batch_size,
        class_mode='categorical', subset='training', seed=CONFIG['seed']
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=target, batch_size=batch_size,
        class_mode='categorical', subset='validation', seed=CONFIG['seed']
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=target, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    n_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())

    print(f"\nJumlah kelas   : {n_classes}")
    print(f"Nama kelas     : {class_names}")
    print(f"Training sampel: {train_gen.samples}")
    print(f"Validasi sampel: {val_gen.samples}")
    print(f"Test sampel    : {test_gen.samples}\n")

    return train_gen, val_gen, test_gen, n_classes, class_names


# ─────────────────────────────────────────────
# 2. Arsitektur Model
# ─────────────────────────────────────────────

def build_cnn_scratch(input_shape, n_classes, dropout=0.4):
    """
    CNN dilatih dari nol sebagai baseline.
    Arsitektur: 4 blok Conv-BN-Pool + Dense
    """
    inputs = keras.Input(shape=input_shape)

    # Blok 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Blok 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Blok 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Blok 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='CNN_Scratch')
    return model


def build_mobilenetv2(input_shape, n_classes, dropout=0.4):
    """
    Transfer Learning dengan MobileNetV2 (pretrained ImageNet).
    Phase 1: Freeze base, latih head
    Phase 2: Fine-tune layer atas
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # Freeze awalnya

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='MobileNetV2_TL')
    return model, base


def build_efficientnetb0(input_shape, n_classes, dropout=0.4):
    """
    Transfer Learning dengan EfficientNetB0 (pretrained ImageNet).
    """
    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='EfficientNetB0_TL')
    return model, base


# ─────────────────────────────────────────────
# 3. Training Pipeline
# ─────────────────────────────────────────────

def get_callbacks(model_name, monitor='val_accuracy'):
    """Buat callback standar untuk semua model."""
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    return [
        EarlyStopping(monitor=monitor, patience=7, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor=monitor, factor=0.3, patience=3,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'checkpoints/{model_name}_best.h5',
                        monitor=monitor, save_best_only=True, verbose=0),
        CSVLogger(f'logs/{model_name}_history.csv'),
    ]


def train_cnn_scratch(train_gen, val_gen, n_classes):
    """Latih CNN dari nol."""
    print("\n" + "─"*50)
    print("TRAINING: CNN dari Nol")
    print("─"*50)

    input_shape = (CONFIG['img_size'], CONFIG['img_size'], 3)
    model = build_cnn_scratch(input_shape, n_classes, CONFIG['dropout'])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=get_callbacks('cnn_scratch'),
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n⏱ Training selesai: {elapsed/60:.1f} menit")

    return model, history, elapsed


def train_transfer_learning(train_gen, val_gen, n_classes, model_type='mobilenet'):
    """
    Latih model transfer learning dengan 2 fase:
    Fase 1 - Feature extraction (base frozen)
    Fase 2 - Fine-tuning (beberapa layer teratas dibuka)
    """
    model_name = 'MobileNetV2' if model_type == 'mobilenet' else 'EfficientNetB0'
    print(f"\n" + "─"*50)
    print(f"TRAINING: {model_name} (Transfer Learning)")
    print("─"*50)

    input_shape = (CONFIG['img_size'], CONFIG['img_size'], 3)

    if model_type == 'mobilenet':
        model, base = build_mobilenetv2(input_shape, n_classes, CONFIG['dropout'])
    else:
        model, base = build_efficientnetb0(input_shape, n_classes, CONFIG['dropout'])

    model.summary()

    # ── Fase 1: Feature Extraction ──
    print(f"\n[Fase 1] Feature Extraction — base frozen ({len(base.layers)} layer)")
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    t0 = time.time()
    hist1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['fine_tune_at'],
        callbacks=get_callbacks(f'{model_type}_phase1'),
        verbose=1
    )

    # ── Fase 2: Fine-tuning ──
    print(f"\n[Fase 2] Fine-tuning — membuka {max(1, len(base.layers)//4)} layer teratas")
    base.trainable = True
    # Freeze semua kecuali 25% layer teratas
    fine_tune_from = len(base.layers) - max(1, len(base.layers) // 4)
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  Layer yang dilatih: {trainable_count}/{len(base.layers)}")

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG['fine_tune_lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'] - CONFIG['fine_tune_at'],
        callbacks=get_callbacks(f'{model_type}_phase2'),
        verbose=1
    )
    elapsed = time.time() - t0

    # Gabungkan history
    combined_history = {}
    for key in hist1.history:
        combined_history[key] = hist1.history[key] + hist2.history[key]

    print(f"\nTraining selesai: {elapsed/60:.1f} menit")
    return model, combined_history, elapsed


# ─────────────────────────────────────────────
# 4. Evaluasi & Error Analysis
# ─────────────────────────────────────────────

def evaluate_model(model, test_gen, class_names, model_name):
    """Evaluasi model pada test set, termasuk analisis error."""
    print(f"\n── Evaluasi: {model_name} ──")

    test_gen.reset()
    y_pred_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes[:len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    # Error analysis: temukan kasus sulit
    errors_idx = np.where(y_pred != y_true)[0]
    error_pairs = {}
    for idx in errors_idx:
        pair = (class_names[y_true[idx]], class_names[y_pred[idx]])
        error_pairs[pair] = error_pairs.get(pair, 0) + 1

    sorted_errors = sorted(error_pairs.items(), key=lambda x: -x[1])
    print(f"\nTop 5 Kasus Sulit (pasangan kelas yang sering salah):")
    for (true_c, pred_c), count in sorted_errors[:5]:
        print(f"  {true_c:15s} → {pred_c:15s}: {count} error")

    # Confidence analisis
    correct_conf = y_pred_prob[np.arange(len(y_pred)), y_pred][y_pred == y_true]
    wrong_conf   = y_pred_prob[np.arange(len(y_pred)), y_pred][y_pred != y_true]
    print(f"\nMean confidence — benar: {correct_conf.mean():.3f} | salah: {wrong_conf.mean():.3f}")

    return acc, cm, y_pred, y_true, y_pred_prob, sorted_errors


# ─────────────────────────────────────────────
# 5. Visualisasi
# ─────────────────────────────────────────────

def plot_training_history(histories_dict):
    """Plot training history semua model."""
    n_models = len(histories_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    fig.suptitle('Training History — Semua Model', fontsize=14, fontweight='bold')

    for col, (name, hist) in enumerate(histories_dict.items()):
        # Accuracy
        axes[0, col].plot(hist['accuracy'],     label='Train', color='steelblue')
        axes[0, col].plot(hist['val_accuracy'],  label='Val',   color='tomato')
        axes[0, col].set_title(f'{name}\nAccuracy', fontsize=11)
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Accuracy')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        # Loss
        axes[1, col].plot(hist['loss'],     label='Train', color='steelblue')
        axes[1, col].plot(hist['val_loss'], label='Val',   color='tomato')
        axes[1, col].set_title(f'{name}\nLoss', fontsize=11)
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('Loss')
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tugas3_training_history.png', dpi=150, bbox_inches='tight')
    print("Disimpan: tugas3_training_history.png")
    plt.show()


def plot_comparison_dashboard(results_dict, class_names):
    """Dashboard perbandingan semua model."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Tugas 3: Transfer Learning vs CNN dari Nol\nDataset: Intel Image Classification',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    model_names = list(results_dict.keys())
    accs = [results_dict[m]['acc'] for m in model_names]
    times = [results_dict[m]['time']/60 for m in model_names]

    colors = ['#4C72B0', '#DD8452', '#55A868']

    # ── Plot 1: Akurasi perbandingan ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(model_names, accs, color=colors[:len(model_names)],
                   edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Perbandingan Akurasi\nTest Set', fontsize=12)
    ax1.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Plot 2: Waktu training ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(model_names, times, color=colors[:len(model_names)], edgecolor='black', alpha=0.85)
    ax2.set_xlabel('Waktu (menit)', fontsize=12)
    ax2.set_title('Waktu Training', fontsize=12)
    for i, t in enumerate(times):
        ax2.text(t + 0.1, i, f'{t:.1f} min', va='center', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')

    # ── Plot 3: Akurasi vs Waktu (trade-off) ──
    ax3 = fig.add_subplot(gs[0, 2])
    for i, name in enumerate(model_names):
        ax3.scatter(times[i], accs[i], s=200, color=colors[i], label=name,
                    zorder=5, edgecolors='black')
        ax3.annotate(name, (times[i], accs[i]),
                     textcoords='offset points', xytext=(8, 4), fontsize=9)
    ax3.set_xlabel('Waktu Training (menit)', fontsize=12)
    ax3.set_ylabel('Test Accuracy', fontsize=12)
    ax3.set_title('Trade-off: Akurasi vs Waktu', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4, 5, 6: Confusion Matrix ──
    for col, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[1, col])
        cm = results_dict[model_name]['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=False, annot_kws={'size': 9})
        ax.set_title(f'{model_name}\n(Norm. CM)', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.tick_params(axis='x', rotation=40, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.savefig('tugas3_comparison.png', dpi=150, bbox_inches='tight')
    print("Disimpan: tugas3_comparison.png")
    plt.show()


def plot_error_analysis(test_gen, model, class_names, model_name, y_pred, y_true, y_pred_prob, n_show=12):
    """Visualisasi gambar yang salah diklasifikasi (kasus sulit)."""
    filenames = test_gen.filenames
    error_idx = np.where(np.array(y_pred) != np.array(y_true))[0]

    if len(error_idx) == 0:
        print("Tidak ada error — akurasi 100%!")
        return

    # Sort by confidence (paling yakin tapi salah → paling menarik)
    error_conf = y_pred_prob[error_idx, y_pred[error_idx]]
    sorted_errors = error_idx[np.argsort(-error_conf)][:n_show]

    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'Error Analysis — {model_name}\n(Gambar yang Salah Diklasifikasi, Diurutkan by Confidence)',
                 fontsize=13, fontweight='bold')

    from tensorflow.keras.preprocessing import image as keras_image

    for i, idx in enumerate(sorted_errors):
        ax = axes[i // n_cols][i % n_cols]
        try:
            fpath = os.path.join(test_gen.directory, filenames[idx])
            img = keras_image.load_img(fpath, target_size=(CONFIG['img_size'], CONFIG['img_size']))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)

        conf = y_pred_prob[idx, y_pred[idx]]
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        ax.set_title(f'{true_label}\n{pred_label} ({conf:.2%})',
                     fontsize=9, color='darkred')
        ax.axis('off')

    # Kosongkan sisa subplot
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(f'tugas3_error_analysis_{model_name}.png', dpi=150, bbox_inches='tight')
    print(f"Disimpan: tugas3_error_analysis_{model_name}.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  TUGAS 3: TRANSFER LEARNING UNTUK DATASET KUSTOM")
    print("  Model: MobileNetV2, EfficientNetB0, CNN Scratch")
    print("  Dataset: Intel Image Classification (6 kelas)")
    print("=" * 60 + "\n")

    # ── Cek ketersediaan dataset ──
    if not os.path.exists(CONFIG['dataset_dir']):
        print("Dataset belum tersedia!")
        print("   Download dataset Intel Image Classification dari Kaggle:")
        print("   https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
        print("\n   Cara download dengan Kaggle CLI:")
        print("   pip install kaggle")
        print("   kaggle datasets download -d puneet6060/intel-image-classification")
        print("   unzip intel-image-classification.zip -d ./dataset")
        print("\n   Setelah download, jalankan kembali script ini.")
        return

    # ── 1. Persiapan data ──
    train_gen, val_gen, test_gen, n_classes, class_names = prepare_data_generators(
        CONFIG['dataset_dir'], CONFIG['img_size'], CONFIG['batch_size']
    )

    input_shape = (CONFIG['img_size'], CONFIG['img_size'], 3)
    histories   = {}
    results     = {}

    # ── 2. Training CNN dari Nol ──
    model_cnn, hist_cnn, time_cnn = train_cnn_scratch(train_gen, val_gen, n_classes)
    histories['CNN Scratch'] = hist_cnn.history

    # ── 3. Training MobileNetV2 ──
    model_mob, hist_mob, time_mob = train_transfer_learning(
        train_gen, val_gen, n_classes, model_type='mobilenet'
    )
    histories['MobileNetV2'] = hist_mob

    # ── 4. Training EfficientNetB0 ──
    model_eff, hist_eff, time_eff = train_transfer_learning(
        train_gen, val_gen, n_classes, model_type='efficientnet'
    )
    histories['EfficientNetB0'] = hist_eff

    # ── 5. Evaluasi ──
    print("\n" + "=" * 60)
    print("EVALUASI SEMUA MODEL PADA TEST SET")
    print("=" * 60)

    for model_name, model, elapsed in [
        ('CNN Scratch',   model_cnn, time_cnn),
        ('MobileNetV2',   model_mob, time_mob),
        ('EfficientNetB0',model_eff, time_eff),
    ]:
        acc, cm, y_pred, y_true, y_pred_prob, errors = evaluate_model(
            model, test_gen, class_names, model_name
        )
        results[model_name] = {
            'acc'    : acc,
            'cm'     : cm,
            'y_pred' : y_pred,
            'y_true' : y_true,
            'prob'   : y_pred_prob,
            'errors' : errors,
            'time'   : elapsed,
        }

    # ── 6. Visualisasi ──
    plot_training_history(histories)
    plot_comparison_dashboard(results, class_names)

    # Error analysis pada model terbaik
    best_model_name = max(results, key=lambda k: results[k]['acc'])
    best_model_obj  = {
        'CNN Scratch'   : model_cnn,
        'MobileNetV2'   : model_mob,
        'EfficientNetB0': model_eff,
    }[best_model_name]

    print(f"\nAnalisis error pada model terbaik: {best_model_name}")
    plot_error_analysis(
        test_gen, best_model_obj, class_names, best_model_name,
        results[best_model_name]['y_pred'],
        results[best_model_name]['y_true'],
        results[best_model_name]['prob'],
    )

    # ── 7. Simpan hasil ──
    summary = {m: {'accuracy': float(v['acc']), 'training_time_min': float(v['time']/60)}
               for m, v in results.items()}
    with open('tugas3_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── 8. Ringkasan ──
    print("\n" + "=" * 60)
    print("RINGKASAN AKHIR")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['acc'])
    for rank, (name, val) in enumerate(sorted_results, 1):
        print(f"#{rank} {name:<18} Acc={val['acc']:.4f} | Waktu={val['time']/60:.1f} min")

    print(f"\nModel Terbaik: {sorted_results[0][0]}")
    print(f"   Test Accuracy : {sorted_results[0][1]['acc']:.4f}")
    print(f"   Waktu Training: {sorted_results[0][1]['time']/60:.1f} menit")


if __name__ == "__main__":
    main()