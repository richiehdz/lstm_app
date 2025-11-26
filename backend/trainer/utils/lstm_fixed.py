# trainer/utils/lstm_fixed.py
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ⚙️ Configuración fija (ajusta estos valores exactos según tu notebook)
PASOS = 3  # <-- aquí pones el número de pasos que usas en el notebook
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

def crear_secuencias(vals: np.ndarray, pasos: int):
    """
    Copia aquí la implementación que tienes en el notebook.
    Debe devolver X (secuencias) e y (target para cada secuencia).
    """
    X, y = [], []
    for i in range(len(vals) - pasos):
        ventana = vals[i:i+pasos, :]
        target = vals[i+pasos, -1]  # asumiendo que la última col es TARGET
        X.append(ventana)
        y.append(target)
    return np.array(X), np.array(y)


def preparar_datos(df: pd.DataFrame):
    # Limpieza mínima
    df = df.copy()
    df = df.dropna()  # igual que en tu notebook
    df["Materia"] = df["Materia"].astype(str).str.strip().str.upper()

    # LabelEncoder de Materia
    le = LabelEncoder()
    df["materia_codificada"] = le.fit_transform(df["Materia"])
    dic_materia = dict(zip(df["materia_codificada"], df["Materia"]))

    # Escalado
    scaler = StandardScaler()
    df[COLS_ESCALAR] = scaler.fit_transform(df[COLS_ESCALAR])

    # Construir secuencias por materia_codificada
    X_total, y_total = [], []
    for mid in df["materia_codificada"].unique():
        mdf = df[df["materia_codificada"] == mid].sort_values("semestre_numerico")
        vals = mdf[FEATURES + [TARGET]].values
        if len(vals) > PASOS:
            Xs, ys = crear_secuencias(vals, PASOS)
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
    # Como en tu notebook: tomar semestre_max, semestre_pred,
    # armar ventana de tamaño PASOS por materia, modificar semestre_numerico,
    # pasar por el modelo y desescalar y.
    # Aquí solo dejo el esqueleto; puedes copiar literalmente la lógica del notebook.

    resultados = []

    semestre_max  = int(df["semestre_numerico"].max())
    semestre_pred = semestre_max + 1

    # Obtener media y std de la columna TARGET ya escalada
    # (si quieres replicar exactamente tu forma de desescalar)
    # o puedes usar scaler.inverse_transform si separas la columna.
    # Aquí asumo que TARGET está en COLS_ESCALAR y usas scaler.mean_/scale_.

    for mid in df["materia_codificada"].unique():
        mdf = df[df["materia_codificada"] == mid].sort_values("semestre_numerico")
        if len(mdf) >= PASOS:
            ultima_ventana = mdf[FEATURES].values[-PASOS:]
            # Modificar semestre_numerico al semestre_pred en la última fila
            nueva_fila = ultima_ventana[-1].copy()
            idx_sem = FEATURES.index("semestre_numerico")
            nueva_fila[idx_sem] = semestre_pred
            ultima_ventana_mod = np.vstack([ultima_ventana[:-1], nueva_fila])

            X_pred = ultima_ventana_mod[np.newaxis, :, :]  # (1, PASOS, feat_dim)
            y_pred_norm = model.predict(X_pred)[0, 0]

            # Aquí desescalas igual que en el notebook
            # y_pred = y_pred_norm * std_y + mean_y

            resultados.append({
                "Materia": dic_materia[mid],
                "materia_codificada": mid,
                "semestre_pred": semestre_pred,
                "cupos_usados_estimados": float(y_pred_norm),  # luego ajustas desescalado
            })

    df_pred = pd.DataFrame(resultados)
    return df_pred
