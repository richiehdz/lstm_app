import pandas as pd
import re


def procesar_semestre_desde_nombre(nombre_archivo: str):
    """
    Extrae year + ciclo (A/B) desde nombres del tipo: ...2020A..., 2019B... etc.
    Devuelve (sem_num, sem_ori) o None si no coincide.
    """
    m = re.search(r"(\d{4})([AB])", nombre_archivo)
    if not m:
        return None
    year, ciclo = m.groups()
    sem_num = int(year) * 2 + (0 if ciclo.upper() == "A" else 1)
    sem_ori = f"{year}{ciclo.upper()}"
    return sem_num, sem_ori


def unificar_resumidos(
    archivos_resumidos: list,  # lista de pares (nombre_archivo, DataFrame)
    archivo_ingresos: tuple | None = None  # (nombre, DataFrame) opcional
):
    """
    Reimplementa tu script original pero trabajando EN MEMORIA.
    Recibe una lista de DataFrames ya cargados.

    Devuelve un DataFrame final ya unificado y limpio.
    """
    partes = []

    for nombre, df in archivos_resumidos:
        # 1. Obtener semestre desde el nombre
        sem_info = procesar_semestre_desde_nombre(nombre)
        if not sem_info:
            print(f"⚠️ Se omite archivo mal nombrado: {nombre}")
            continue

        sem_num, sem_ori = sem_info

        # 2. Validar columnas esperadas
        req = {"Materia", "Total_Cupos", "Total_Secciones", "Residuos_Cupos"}
        faltantes = req - set(df.columns)
        if faltantes:
            raise ValueError(f"{nombre} sin columnas {faltantes}")

        # 3. Limpieza como en script
        df = df.copy()
        df["Materia"] = df["Materia"].astype(str).str.strip().str.upper()

        for c in ["Total_Cupos", "Total_Secciones", "Residuos_Cupos"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["semestre_numerico"] = sem_num
        df["semestre_original"] = sem_ori

        partes.append(df)

    if not partes:
        raise ValueError("No se pudo procesar ningún archivo válido.")

    # 4. Unir todo
    df_total = pd.concat(partes, ignore_index=True)

    # 5. Cupos usados
    df_total["cupos_usados"] = (
        df_total["Total_Cupos"].fillna(0) - df_total["Residuos_Cupos"].fillna(0)
    ).clip(lower=0)

    # 6. Merge con ingresos (si se proporcionó)
    if archivo_ingresos:
        nombre_ing, ing = archivo_ingresos

        # Renombrado flexible de columnas
        rename_map = {}
        for c in ing.columns:
            cl = str(c).lower()
            if cl.startswith("semestre"):
                rename_map[c] = "Semestre"
            elif "ingres" in cl:
                rename_map[c] = "Ingresados"

        ing = ing.rename(columns=rename_map)

        if not {"Semestre", "Ingresados"} <= set(ing.columns):
            raise ValueError(
                f"El archivo '{nombre_ing}' debe tener columnas 'Semestre' y 'Ingresados'."
            )

        ing["Semestre"] = ing["Semestre"].astype(str).str.strip().str.upper()
        ing["Ingresados"] = (
            pd.to_numeric(ing["Ingresados"], errors="coerce").fillna(0).astype(int)
        )

        ing = ing.groupby("Semestre", as_index=False)["Ingresados"].sum()

        df_total["semestre_original"] = (
            df_total["semestre_original"].astype(str).str.strip().str.upper()
        )

        df_total = (
            df_total.merge(
                ing, left_on="semestre_original", right_on="Semestre", how="left"
            )
            .drop(columns=["Semestre"])
            .rename(columns={"Ingresados": "nuevos_alumnos"})
        )

        df_total["nuevos_alumnos"] = df_total["nuevos_alumnos"].fillna(0).astype(int)
    else:
        print("⚠️ No se recibió archivo de ingresos. 'nuevos_alumnos'=0.")
        df_total["nuevos_alumnos"] = 0

    # 7. Orden y lag1 por materia
    df_total = df_total.sort_values(["Materia", "semestre_numerico"]).reset_index(
        drop=True
    )
    df_total["lag1_cupos_usados"] = (
        df_total.groupby("Materia")["cupos_usados"].shift(1).fillna(0)
    )

    # 8. Columnas finales
    keep = [
        "Materia",
        "Total_Cupos",
        "Total_Secciones",
        "Residuos_Cupos",
        "semestre_numerico",
        "semestre_original",
        "cupos_usados",
        "nuevos_alumnos",
        "lag1_cupos_usados",
    ]

    faltantes_final = [c for c in keep if c not in df_total.columns]
    if faltantes_final:
        raise ValueError(f"Faltan columnas al final: {faltantes_final}")

    return df_total[keep]
