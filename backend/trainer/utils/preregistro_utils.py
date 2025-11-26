import re
import pandas as pd
import numpy as np


def semestre_to_numeric(sem):
    """
    2024A -> 2024*2
    2024B -> 2024*2 + 1
    """
    if pd.isna(sem):
        return None
    m = re.match(r"^\s*(\d{4})\s*([ABab])\s*$", str(sem))
    if not m:
        return None
    year = int(m.group(1))
    letter = m.group(2).upper()
    return year * 2 + (1 if letter == "B" else 0)


def normalize_preregistro_df(df: pd.DataFrame, semestre_original: str) -> pd.DataFrame:
    """
    Replica la lógica de normalize_preregistro_df de tu script original. :contentReference[oaicite:2]{index=2}
    """
    cols = {c.strip().upper(): c for c in df.columns}

    def get_col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    col_materia = get_col("MATERIA", "Materia")
    col_reg = get_col("REGISTRADOS", "REGISTRADO", "REG")
    # OJO: TURNO ≡ PREREGISTRADOS
    col_turno = get_col("TURNO", "PREREGISTRO", "PREREGISTRADOS", "PREREG")

    out = pd.DataFrame()
    out["Materia"] = df[col_materia].astype(str).str.strip().str.upper()
    out["registrados"] = pd.to_numeric(df[col_reg], errors="coerce")
    out["preregistrados"] = pd.to_numeric(df[col_turno], errors="coerce")
    out["semestre_original"] = semestre_original
    out["semestre_numerico"] = semestre_to_numeric(semestre_original)
    return out


def combinar_oferta_y_preregistro(oferta_df: pd.DataFrame, preregistros: list[pd.DataFrame], nombres: list[str]):
    """
    Implementa la lógica de unifier.py pero en memoria, recibiendo:
    - oferta_df: DataFrame de oferta_academica_unificada.csv
    - preregistros: lista de DataFrames de preregistros
    - nombres: lista de nombres de archivo (para extraer semestre)

    Devuelve:
    - merged (oferta_con_preregistro)
    - pr_agg (preregistro_unificado_consolidado)
    """
    # Normalizar Materia de oferta
    oferta = oferta_df.copy()
    oferta["Materia"] = oferta["Materia"].astype(str).str.strip().str.upper()

    frames = []
    for df, fname in zip(preregistros, nombres):
        # Extraer semestre desde el nombre: 2018B_preregistro_filtrado.xlsx
        m = re.match(r"^(\d{4}[ABab])_preregistro_filtrado\.", fname)
        semestre_original = m.group(1) if m else None
        frames.append(normalize_preregistro_df(df, semestre_original))

    pr_all = pd.concat(frames, ignore_index=True)

    # Consolidar preregistro (llave Materia + semestre) :contentReference[oaicite:3]{index=3}
    keys = ["Materia", "semestre_original", "semestre_numerico"]
    pr_agg = (
        pr_all
        .groupby(keys, as_index=False)[["registrados", "preregistrados"]]
        .sum(min_count=1)
    )

    # Merge LEFT: no se crean filas nuevas
    before_rows = len(oferta)
    merged = oferta.merge(pr_agg, on=keys, how="left")
    after_rows = len(merged)
    if before_rows != after_rows:
        raise ValueError("El número de filas cambió tras el merge, y no debería.")

    return merged, pr_agg


def imputar_promedios_ab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implementa la lógica de imputar_ab_means.py: rellenar registrados y preregistrados
    por promedio de (Materia, calendario A/B). :contentReference[oaicite:4]{index=4}
    """
    df = df.copy()
    df["Materia"] = df["Materia"].astype(str).str.strip().str.upper()

    # Extraer calendario (A/B) desde semestre_original (ej: 2025B -> B)
    df["calendario"] = (
        df["semestre_original"].astype(str).str.strip().str[-1].str.upper()
    )
    df.loc[~df["calendario"].isin(["A", "B"]), "calendario"] = np.nan

    # Calcular medias por (Materia, calendario)
    means = (
        df.groupby(["Materia", "calendario"])[["registrados", "preregistrados"]]
          .mean()
          .rename(columns={
              "registrados": "mean_registrados",
              "preregistrados": "mean_preregistrados"
          })
    )

    df = df.merge(means, on=["Materia", "calendario"], how="left")

    # Rellenar solo NaN donde sí hay media
    for col, mean_col in [
        ("registrados", "mean_registrados"),
        ("preregistrados", "mean_preregistrados"),
    ]:
        mask = df[col].isna() & df[mean_col].notna()
        df.loc[mask, col] = df.loc[mask, mean_col].round()

    # Convertir a enteros "nullable" (para conservar vacíos no imputables)
    for col in ["registrados", "preregistrados"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    df = df.drop(columns=["mean_registrados", "mean_preregistrados"], errors="ignore")
    return df
