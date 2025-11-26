# backend/trainer/utils/resumir_datasets.py
import pandas as pd

MATERIAS_EXCLUIDAS = [
    "PROYECTO DE GESTION DE LA TECNOLOGIA DE INFORMACION",
    "PROYECTO DE SISTEMAS ROBUSTOS, PARALELOS Y DISTRIBUIDOS",
    "COMPUTO FLEXIBLE (SOFTCOMPUTING)",
    "ANALISIS DE PROBLEMAS GLOBALES DEL SIGLO XXI",
]


def resumir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la misma lógica que tu script original, pero sobre un DataFrame.
    Devuelve un nuevo DataFrame resumido.
    """
    # Normalizar Materia como en el script original
    df = df.copy()
    df["Materia"] = df["Materia"].astype(str).str.strip().str.upper()

    clases = [
        m
        for m in df["Materia"].dropna().unique()
        if m not in MATERIAS_EXCLUIDAS
    ]

    # Secciones por materia
    secciones = (
        df.groupby("Materia")["Sec"]
        .nunique()
        .reindex(clases)
        .fillna(0)
        .astype(int)
    )

    # Cupos / Residuos por materia
    total_cupos = (
        df.groupby("Materia")["CUP"].sum().reindex(clases).fillna(0)
    )
    residuos = (
        df.groupby("Materia")["DIS"].sum().reindex(clases).fillna(0)
    )

    df_resumen = pd.DataFrame(
        {
            "Materia": clases,
            "Total_Cupos": total_cupos.values,
            "Total_Secciones": secciones.values,
            "Residuos_Cupos": residuos.values,
        }
    )
    return df_resumen
