from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from io import BytesIO
import zipfile
import pandas as pd
from .utils.union_datasets import unificar_resumidos
from django.shortcuts import render
from .forms import UploadDatasetForm, MultiCSVUploadForm
from .tasks import train_lstm_job
from .utils.resumir_datasets import resumir_dataframe
from .models import Dataset, TrainingJob, IngresoSemestral
from .utils.preregistro_utils import (
    combinar_oferta_y_preregistro,
    imputar_promedios_ab,
)

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