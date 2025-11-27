from django.shortcuts import render, redirect, get_object_or_404
from io import BytesIO
from django.views.decorators.http import require_GET
import zipfile
import os
import io
import csv
import pandas as pd
from .utils.union_datasets import unificar_resumidos
from django.shortcuts import render
from .forms import UploadDatasetForm, MultiCSVUploadForm
from .tasks import train_lstm_job
from .utils.resumir_datasets import resumir_dataframe
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404, HttpResponse
from rest_framework.views import APIView
from rest_framework import status
from .models import Dataset, TrainingJob, IngresoSemestral
from .utils.preregistro_utils import (
    combinar_oferta_y_preregistro,
    imputar_promedios_ab,
)
from django.contrib import messages

from .forms import CompararPrediccionesForm

def upload_and_train(request):
    if request.method == "POST":
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data["file"]

            # 1) Crear Dataset
            dataset = Dataset.objects.create(
                original_name=uploaded_file.name,
                file=uploaded_file,
            )

            # 2) Analizar CSV de forma básica
            df = pd.read_csv(dataset.file.path)
            dataset.n_rows = df.shape[0]
            dataset.n_cols = df.shape[1]
            dataset.columns_schema = [
                {"name": col, "dtype": str(df[col].dtype)}
                for col in df.columns
            ]
            dataset.save()

            # 3) Crear TrainingJob
            input_cols = [
                c.strip()
                for c in request.POST.get("input_columns", "").split(",")
                if c.strip()
            ]

            job = TrainingJob.objects.create(
                dataset=dataset,
                target_column=form.cleaned_data["target_column"],
                input_columns=input_cols,
                window_size=form.cleaned_data["window_size"],
                epochs=form.cleaned_data["epochs"],
            )

            # 4) Lanzar tarea Celery dummy
            train_lstm_job.delay(str(job.id))

            return redirect("trainer:job_detail", job_id=job.id)
    else:
        form = UploadDatasetForm()

    return render(request, "trainer/upload_and_train.html", {"form": form})


def job_detail(request, job_id):
    job = get_object_or_404(TrainingJob, id=job_id)
    return render(request, "trainer/job_detail.html", {"job": job})

'''def leer_tabular(uploaded_file):
    """
    Lee un archivo subido (CSV o Excel) y devuelve un DataFrame.
    Intenta primero como CSV UTF-8 y, si falla por codificación,
    reintenta con latin-1. Para .xlsx/.xls usa read_excel.
    """
    nombre = getattr(uploaded_file, "name", "").lower()

    # Si es Excel, usamos read_excel directamente
    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    # Caso general: CSV u otra extensión de texto
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # Volvemos el puntero al inicio y reintentamos con otra codificación
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")'''
def leer_tabular(uploaded_file):
    """
    Lee un archivo subido (CSV o Excel) y devuelve un DataFrame.
    - Para .xlsx/.xls usa read_excel.
    - Para .csv deja que pandas detecte el separador (',' o ';').
    """
    nombre = getattr(uploaded_file, "name", "").lower()

    # ✅ Si es Excel
    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    # ✅ Si es CSV u otro texto: intentar detectar el separador
    try:
        # Deja que pandas use csv.Sniffer para inferir ',' vs ';', etc.
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        # Fallback explícito a ';'
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";", engine="python")
        except Exception:
            # Último intento con los defaults (que lanzará el error real si falla)
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)


def resumir_datasets_originales_view(request):
    """
    Sube varios CSV/XLSX, aplica la lógica de resumen y devuelve un ZIP
    con los archivos resumidos.
    """
    if request.method == "POST":
        form = MultiCSVUploadForm(request.POST)

        if form.is_valid():
            files = request.FILES.getlist("files")  # 👈 nombre del input en el template
            max_files = 30

            if not files:
                # Error manual si no se mandaron archivos
                return render(
                    request,
                    "trainer/resumir_dataset_originales.html",
                    {
                        "form": form,
                        "error": "Debes seleccionar al menos un archivo.",
                    },
                )

            if len(files) > max_files:
                return render(
                    request,
                    "trainer/resumir_dataset_originales.html",
                    {
                        "form": form,
                        "error": f"Solo se permiten hasta {max_files} archivos a la vez.",
                    },
                )

            allowed_exts = (".csv", ".xlsx", ".xls")

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for f in files:
                    filename = f.name.lower()
                    if not any(filename.endswith(ext) for ext in allowed_exts):
                        # Saltamos archivos con extensión inválida
                        continue

                    # Leer según el tipo
                    if filename.endswith(".csv"):
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_excel(f)

                    # Aplicar lógica de resumen
                    df_resumen = resumir_dataframe(df)

                    # Guardar cada resumen como CSV dentro del ZIP
                    output_buffer = BytesIO()
                    nombre_base = f.name.rsplit(".", 1)[0]
                    nombre_csv = f"resumen_cupos_{nombre_base}.csv"

                    df_resumen.to_csv(output_buffer, index=False)
                    output_buffer.seek(0)

                    zip_file.writestr(nombre_csv, output_buffer.getvalue())

            zip_buffer.seek(0)
            response = HttpResponse(
                zip_buffer.getvalue(),
                content_type="application/zip",
            )
            response["Content-Disposition"] = (
                'attachment; filename="resumenes_cupos.zip"'
            )
            return response
    else:
        form = MultiCSVUploadForm()

    return render(
        request,
        "trainer/resumir_dataset_originales.html",
        {"form": form},
    )


def unir_datasets_resumidos_view(request):
    """
    Sube varios datasets resumidos y los unifica.
    Los datos de ingresos se toman de la tabla IngresoSemestral.
    Devuelve un CSV con la oferta académica unificada.
    """
    if request.method == "POST":
        archivos = request.FILES.getlist("resumidos")

        if not archivos:
            return render(
                request,
                "trainer/unir_datasets_resumidos.html",
                {"error": "Debes subir al menos un archivo resumido."},
            )

        # 1) Leer datasets resumidos a memoria
        lista_resumidos = []
        for f in archivos:
            nombre = f.name
            if nombre.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            lista_resumidos.append((nombre, df))

        # 2) Construir DataFrame de ingresos desde la BD
        qs = IngresoSemestral.objects.all()
        data_ingresos = None
        if qs.exists():
            df_ing = pd.DataFrame(list(qs.values("semestre", "ingresados")))
            # Ajustamos nombres a los que espera el helper ('Semestre', 'Ingresados')
            df_ing = df_ing.rename(
                columns={"semestre": "Semestre", "ingresados": "Ingresados"}
            )
            data_ingresos = ("ingresos_db", df_ing)
        else:
            data_ingresos = None  # nuevos_alumnos=0 dentro de unificar_resumidos

        # 3) Ejecutar unificación con la lógica de union_datasets
        df_final = unificar_resumidos(lista_resumidos, data_ingresos)

        # 4) Generar CSV en memoria para descarga
        buffer = BytesIO()
        df_final.to_csv(buffer, index=False)
        buffer.seek(0)

        response = HttpResponse(
            buffer.getvalue(),
            content_type="text/csv",
        )
        response["Content-Disposition"] = (
            'attachment; filename="oferta_academica_unificada.csv"'
        )
        return response

    return render(request, "trainer/unir_datasets_resumidos.html")


def combinar_oferta_y_preregistro_view(request):
    """
    Vista que:
    1) Combina oferta_academica_unificada.csv con varios archivos *_preregistro_filtrado.xlsx
       -> genera oferta_con_preregistro y preregistro_unificado_consolidado.
    2) Aplica imputación de promedios A/B sobre registrados y preregistrados
       -> genera oferta_con_preregistro_imputada.
    3) Devuelve todo en un ZIP descargable.
    """
    if request.method == "POST":
        oferta_file = request.FILES.get("oferta")
        prereg_files = request.FILES.getlist("preregistros")

        if not oferta_file:
            return render(
                request,
                "trainer/combinar_preregistro.html",
                {"error": "Debes subir el archivo oferta_academica_unificada.csv."},
            )

        if not prereg_files:
            return render(
                request,
                "trainer/combinar_preregistro.html",
                {"error": "Debes subir al menos un archivo *_preregistro_filtrado.xlsx."},
            )

        # Leer oferta
        try:
            oferta_df = pd.read_csv(oferta_file)
        except Exception as e:
            return render(
                request,
                "trainer/combinar_preregistro.html",
                {"error": f"Error leyendo la oferta: {e}"},
            )

        # Leer preregistros
        prereg_dfs = []
        nombres = []
        for f in prereg_files:
            fname = f.name
            try:
                if fname.lower().endswith(".csv"):
                    df_pr = pd.read_csv(f)
                else:
                    df_pr = pd.read_excel(f)
            except Exception as e:
                return render(
                    request,
                    "trainer/combinar_preregistro.html",
                    {"error": f"Error leyendo '{fname}': {e}"},
                )
            prereg_dfs.append(df_pr)
            nombres.append(fname)

        # Paso 1: combinar oferta + preregistros (unifier.py)
        try:
            oferta_con_pr, pr_agg = combinar_oferta_y_preregistro(
                oferta_df, prereg_dfs, nombres
            )
        except Exception as e:
            return render(
                request,
                "trainer/combinar_preregistro.html",
                {"error": f"Error al combinar oferta y preregistros: {e}"},
            )

        # Paso 2: imputar promedios A/B (promedio.py)
        try:
            oferta_imputada = imputar_promedios_ab(oferta_con_pr)
        except Exception as e:
            return render(
                request,
                "trainer/combinar_preregistro.html",
                {"error": f"Error al imputar promedios: {e}"},
            )
        for df in (oferta_con_pr, pr_agg, oferta_imputada):
            df["Materia"] = df["Materia"].astype(str).str.strip().str.upper()
            df.drop(df[df["Materia"] == "NAN"].index, inplace=True)
        # Armar ZIP en memoria
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 1) oferta_con_preregistro.csv
            buf1 = BytesIO()
            oferta_con_pr.to_csv(buf1, index=False)
            buf1.seek(0)
            zip_file.writestr("oferta_con_preregistro.csv", buf1.getvalue())

            # 2) preregistro_unificado_consolidado.csv
            buf2 = BytesIO()
            pr_agg.to_csv(buf2, index=False)
            buf2.seek(0)
            zip_file.writestr("preregistro_unificado_consolidado.csv", buf2.getvalue())

            # 3) oferta_con_preregistro_imputada.csv
            buf3 = BytesIO()
            oferta_imputada.to_csv(buf3, index=False)
            buf3.seek(0)
            zip_file.writestr("oferta_con_preregistro_imputada.csv", buf3.getvalue())

        zip_buffer.seek(0)
        response = HttpResponse(
            zip_buffer.getvalue(),
            content_type="application/zip",
        )
        response["Content-Disposition"] = (
            'attachment; filename="oferta_preregistro_procesada.zip"'
        )
        return response

    # GET: mostrar formulario
    return render(request, "trainer/combinar_preregistro.html")

def home(request):
    return render(request, "trainer/home.html")



class TrainingJobDetailView(APIView):
    """
    GET /api/training-jobs/<id>/
    Devuelve el estado del job, progreso y si hay archivo listo.
    """

    # permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        job = get_object_or_404(TrainingJob, pk=pk)

        return Response(
            {
                "id": str(job.id),
                "status": job.status,
                "progress": getattr(job, "progress", None),
                "output_available": bool(job.output_path),
            },
            status=status.HTTP_200_OK,
        )
class TrainingJobDownloadView(APIView):
    """
    GET /api/training-jobs/<id>/download/
    Devuelve el CSV de predicciones como archivo descargable.
    """

    # permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        job = get_object_or_404(TrainingJob, pk=pk)

        if not job.output_path:
            raise Http404("El archivo de resultados aún no está disponible.")

        if not os.path.exists(job.output_path):
            raise Http404("El archivo de resultados no se encontró en el servidor.")

        filename = os.path.basename(job.output_path)
        file_handle = open(job.output_path, "rb")
        response = FileResponse(
            file_handle,
            as_attachment=True,
            filename=filename,
            content_type="text/csv",
        )
        return response
    


def comparar_predicciones_view(request):
    """
    Vista para subir dos archivos (predicciones vs datos reales),
    emparejar por Materia y calcular MSE/MAE + tabla de comparación.
    """
    contexto = {
        "form": CompararPrediccionesForm(),
        "resultados": None,
        "columnas_usadas": None,
    }

    if request.method == "POST":
        form = CompararPrediccionesForm(request.POST, request.FILES)
        if form.is_valid():
            pred_file = form.cleaned_data["predicciones_file"]
            reales_file = form.cleaned_data["reales_file"]

            try:
                # 1) Leer archivos (CSV o lo que estés usando)
                df_pred = leer_tabular(pred_file)
                df_real = leer_tabular(reales_file)

                if df_pred.empty or df_real.empty:
                    messages.error(request, "Alguno de los archivos está vacío.")
                    return render(request, "trainer/comparar_predicciones.html", contexto)

                # 2) Validar que ambos tengan Materia
                if "Materia" not in df_pred.columns or "Materia" not in df_real.columns:
                    messages.error(
                        request,
                        "Ambos archivos deben tener una columna llamada 'Materia'."
                    )
                    return render(request, "trainer/comparar_predicciones.html", contexto)

                # 3) Elegir columnas de predicción y reales
                # Predicciones: primero intentamos cupos_usados_estimados
                if "cupos_usados_estimados" in df_pred.columns:
                    col_pred = "cupos_usados_estimados"
                else:
                    pred_numeric = df_pred.select_dtypes(include="number")
                    if pred_numeric.shape[1] == 0:
                        messages.error(
                            request,
                            "No se encontraron columnas numéricas en el archivo de predicciones."
                        )
                        return render(request, "trainer/comparar_predicciones.html", contexto)
                    col_pred = pred_numeric.columns[-1]

                # Reales: primero Cupos_Usados_Reales, luego Total_Cupos, y si no, último numérico
                if "Cupos_Usados_Reales" in df_real.columns:
                    col_real = "Cupos_Usados_Reales"
                elif "Total_Cupos" in df_real.columns:
                    col_real = "Total_Cupos"
                else:
                    real_numeric = df_real.select_dtypes(include="number")
                    if real_numeric.shape[1] == 0:
                        messages.error(
                            request,
                            "No se encontraron columnas numéricas en el archivo de datos reales."
                        )
                        return render(request, "trainer/comparar_predicciones.html", contexto)
                    col_real = real_numeric.columns[-1]

                # 4) Normalizar Materia para empatar por texto (evitamos problemas de espacios/minúsculas)
                df_pred["Materia_norm"] = (
                    df_pred["Materia"].astype(str).str.strip().str.upper()
                )
                df_real["Materia_norm"] = (
                    df_real["Materia"].astype(str).str.strip().str.upper()
                )

                # Sub-dataframes con lo que necesitamos
                pred_sub = df_pred[["Materia_norm", "Materia", col_pred]].rename(
                    columns={
                        "Materia": "Materia_pred",
                        col_pred: "prediccion_bruta",
                    }
                )
                real_sub = df_real[["Materia_norm", "Materia", col_real]].rename(
                    columns={
                        "Materia": "Materia_real",
                        col_real: "valor_real",
                    }
                )

                # 5) Hacer merge por Materia_norm
                df_merge = pd.merge(
                    pred_sub,
                    real_sub,
                    on="Materia_norm",
                    how="inner",
                )

                if df_merge.empty:
                    messages.error(
                        request,
                        "No se encontraron materias en común entre predicciones y datos reales."
                    )
                    return render(request, "trainer/comparar_predicciones.html", contexto)

                # Preferimos el nombre de la materia del archivo 'real' si existe
                materias_final = df_merge["Materia_real"].fillna(df_merge["Materia_pred"])

                # 6) Convertir a numérico y aplicar regla: predicción < 0 => 0
                y_real = pd.to_numeric(df_merge["valor_real"], errors="coerce")
                y_pred = pd.to_numeric(df_merge["prediccion_bruta"], errors="coerce")

                # ⚠️ Solo cambiamos las predicciones negativas a 0
                y_pred = y_pred.where(y_pred >= 0, 0)

                # Quitamos filas donde falte real o pred
                mask_valid = y_real.notna() & y_pred.notna()
                y_real = y_real[mask_valid]
                y_pred = y_pred[mask_valid]
                materias_final = materias_final[mask_valid]

                if len(y_real) == 0:
                    messages.error(
                        request,
                        "Después de limpiar datos faltantes no quedó ninguna fila válida para comparar."
                    )
                    return render(request, "trainer/comparar_predicciones.html", contexto)

                # 7) Calcular errores
                diff = y_real - y_pred
                mse = float((diff ** 2).mean())
                mae = float(diff.abs().mean())
                n = len(y_real)

                # 8) Construir DataFrame de comparación por materia
                df_comparacion = pd.DataFrame({
                    "Materia": materias_final.values,
                    "valor_real": y_real.values,
                    "prediccion": y_pred.values,
                    "error": diff.values,
                    "error_absoluto": diff.abs().values,
                    "error_cuadratico": (diff ** 2).values,
                })

                # Guardar CSV en sesión.
                # Usamos ';' como separador para que la coma dentro del nombre de la materia
                # (ej. "USO, ADAPTACION ...") no rompa las columnas al abrir en Excel.
                buffer = io.StringIO()

                # Línea que Excel reconoce como separador
                buffer.write("sep=;\n")

                # Exportar correctamente
                df_comparacion.to_csv(
                    buffer,
                    index=False,
                    sep=";",
                    quoting=csv.QUOTE_ALL
                )

                # Guardar en sesión
                request.session["comparacion_csv"] = buffer.getvalue()

                # Info para mostrar en la pantalla
                contexto["form"] = form
                contexto["resultados"] = {
                    "filas_comparadas": n,
                    "mse": mse,
                    "mae": mae,
                }
                contexto["columnas_usadas"] = {
                    "predicciones": col_pred,
                    "reales": col_real,
                }

                return render(request, "trainer/comparar_predicciones.html", contexto)

            except Exception as e:
                messages.error(request, f"Ocurrió un error al procesar los archivos: {e}")
                return render(request, "trainer/comparar_predicciones.html", contexto)

        # form inválido
        contexto["form"] = form
        return render(request, "trainer/comparar_predicciones.html", contexto)

    # GET
    return render(request, "trainer/comparar_predicciones.html", contexto)





@require_GET
def descargar_comparacion_csv(request):
    csv_data = request.session.get("comparacion_csv")
    if not csv_data:
        raise Http404("No hay resultados de comparación disponibles.")

    response = HttpResponse(csv_data, content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="comparacion_pred_vs_real.csv"'
    return response
