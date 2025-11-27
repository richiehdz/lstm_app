# trainer/utils/lstm_fixed.py
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ⚙️ Configuración fija (ajusta estos valores exactos según tu notebook)
PASOS = 2  # <-- aquí pones el número de pasos que usas en el notebook
EPOCHS = 150
BATCH_SIZE = 32

FEATURES = [
    "materia_codificada",
    "Total_Cupos",
    "Total_Secciones",
    "Residuos_Cupos",
    "semestre_numerico",
    "nuevos_alumnos",
    "lag1_cupos_usados",
    "preregistrados",
]
TARGET = "cupos_usados"

COLS_ESCALAR = [
    "Total_Cupos",
    "Total_Secciones",
    "Residuos_Cupos",
    "semestre_numerico",
    "nuevos_alumnos",
    "lag1_cupos_usados",
    "cupos_usados",
    "preregistrados",
]

def crear_secuencias(feature_vals: np.ndarray, target_vals: np.ndarray, pasos: int):
    """
    feature_vals: np.array de shape (N, n_features) -> SOLO FEATURES
    target_vals:  np.array de shape (N,)           -> SOLO TARGET
    """
    X, y = [], []
    for i in range(len(feature_vals) - pasos):
        ventana = feature_vals[i:i + pasos, :]      # (pasos, n_features)
        target = target_vals[i + pasos]             # valor a predecir
        X.append(ventana)
        y.append(target)
    return np.array(X), np.array(y)



def preparar_datos(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna()
    df["Materia"] = df["Materia"].astype(str).str.strip().str.upper()

    # LabelEncoder de Materia
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder()
    df["materia_codificada"] = le.fit_transform(df["Materia"])
    dic_materia = dict(zip(df["materia_codificada"], df["Materia"]))

    # Escalado
    scaler = StandardScaler()
    df[COLS_ESCALAR] = scaler.fit_transform(df[COLS_ESCALAR])

    X_total, y_total = [], []

    for mid in df["materia_codificada"].unique():
        mdf = df[df["materia_codificada"] == mid].sort_values("semestre_numerico")

        # 👇 SOLO FEATURES en X, y por aparte
        feature_vals = mdf[FEATURES].values           # shape (N, 8)
        target_vals = mdf[TARGET].values             # shape (N,)

        if len(feature_vals) > PASOS:
            Xs, ys = crear_secuencias(feature_vals, target_vals, PASOS)
            if len(Xs):
                X_total.append(Xs)
                y_total.append(ys)

    X = np.vstack(X_total) if X_total else np.empty((0, PASOS, len(FEATURES)))
    y = np.concatenate(y_total) if y_total else np.empty((0,))

    return df, X, y, scaler, le, dic_materia



def construir_modelo(input_shape):
    timesteps, feat_dim = input_shape[0], input_shape[1]

    weights_vec = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 2, 2], dtype=tf.float32)

    model = Sequential([
        layers.Input(shape=(timesteps, feat_dim)),
        layers.Lambda(lambda x: x * weights_vec),
        LSTM(96, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(192, activation="relu"),
        Dense(1, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


def entrenar_modelo(X, y):
    model = construir_modelo((X.shape[1], X.shape[2]))

    callbacks = [EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)]

    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    return model, history


def predecir_siguiente_semestre(df, model, scaler, dic_materia):
    resultados = []

    semestre_max  = int(df["semestre_numerico"].max())
    semestre_pred = semestre_max + 1

    # 1) Sacar índice, media y desviación del TARGET dentro de COLS_ESCALAR
    idx_target_scaler = COLS_ESCALAR.index(TARGET)
    mean_y = scaler.mean_[idx_target_scaler]
    std_y  = scaler.scale_[idx_target_scaler]

    for mid in df["materia_codificada"].unique():
        mdf = df[df["materia_codificada"] == mid].sort_values("semestre_numerico")

        if len(mdf) >= PASOS:
            # Última ventana de FEATURES
            ultima_ventana = mdf[FEATURES].values[-PASOS:]

            # Modificar semestre_numerico al semestre_pred en la última fila
            nueva_fila = ultima_ventana[-1].copy()
            idx_sem = FEATURES.index("semestre_numerico")
            nueva_fila[idx_sem] = semestre_pred

            # Reemplazar la última fila por la nueva
            ultima_ventana_mod = np.vstack([ultima_ventana[:-1], nueva_fila])

            # (1, PASOS, feat_dim)
            X_pred = ultima_ventana_mod[np.newaxis, :, :]

            # 2) Predicción NORMALIZADA (z-score)
            y_pred_norm = model.predict(X_pred)[0, 0]

            # 3) Desescalado a cupos reales
            y_pred_real = y_pred_norm * std_y + mean_y

            resultados.append({
                "Materia": dic_materia[mid],
                "materia_codificada": mid,
                "semestre_pred": semestre_pred,
                "cupos_usados_estimados": float(y_pred_real),
                # Opcional: dejar también el valor normalizado para debug
                "cupos_usados_estimados_normalizado": float(y_pred_norm),
            })

    df_pred = pd.DataFrame(resultados)
    return df_pred
