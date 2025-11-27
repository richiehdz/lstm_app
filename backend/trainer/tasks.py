# trainer/tasks.py
from celery import shared_task
import os
import pandas as pd
from django.utils import timezone
from django.conf import settings

from .models import TrainingJob
from .utils.lstm_fixed import (
    preparar_datos,
    entrenar_modelo,
    predecir_siguiente_semestre,
)


@shared_task(bind=True)
def train_lstm_job(self, job_id: str):
    """
    Tarea Celery para entrenar el modelo LSTM y generar predicciones
    a partir del dataset asociado al TrainingJob.
    """
    try:
        job = TrainingJob.objects.get(id=job_id)

        # Marcar el job como "running"
        job.status = "running"
        if hasattr(job, "progress"):
            job.progress = 0
        if hasattr(job, "started_at"):
            job.started_at = timezone.now()
        job.save(
            update_fields=[
                f
                for f in ["status", "progress", "started_at"]
                if hasattr(job, f)
            ]
        )

        dataset = job.dataset  # relación FK al Dataset

        # 1) Cargar CSV del dataset
        df = pd.read_csv(dataset.file.path)

        if hasattr(job, "progress"):
            job.progress = 20
            job.save(update_fields=["progress"])

        # 2) Preparar datos
        df_proc, X, y, scaler, le, dic_materia = preparar_datos(df)

        if hasattr(job, "progress"):
            job.progress = 50
            job.save(update_fields=["progress"])

        # 3) Entrenar modelo
        model, history = entrenar_modelo(X, y)

        if hasattr(job, "progress"):
            job.progress = 80
            job.save(update_fields=["progress"])

        # 4) Generar predicciones del siguiente semestre
        df_pred = predecir_siguiente_semestre(df_proc, model, scaler, dic_materia)

        # 5) Guardar CSV de predicciones y guardar ruta en job.output_path
        #    -> Lo guardamos en MEDIA_ROOT/training_results/job_<id>_predicciones.csv
        results_dir = os.path.join(settings.MEDIA_ROOT, "training_results")
        os.makedirs(results_dir, exist_ok=True)

        filename = f"job_{job.id}_predicciones.csv"
        out_path = os.path.join(results_dir, filename)

        df_pred.to_csv(out_path, index=False)

        job.status = "done"
        if hasattr(job, "progress"):
            job.progress = 100
        if hasattr(job, "output_path"):
            job.output_path = out_path
        if hasattr(job, "finished_at"):
            job.finished_at = timezone.now()

        job.save(
            update_fields=[
                f
                for f in ["status", "progress", "output_path", "finished_at"]
                if hasattr(job, f)
            ]
        )

        return {
            "status": "ok",
            "job_id": job_id,
            "output_path": getattr(job, "output_path", None),
        }

    except Exception as e:
        # Si algo falla, marcamos el job como "failed"
        try:
            job = TrainingJob.objects.get(id=job_id)
            job.status = "failed"
            if hasattr(job, "progress"):
                job.progress = 100
            if hasattr(job, "finished_at"):
                job.finished_at = timezone.now()

            job.save(
                update_fields=[
                    f
                    for f in ["status", "progress", "finished_at"]
                    if hasattr(job, f)
                ]
            )
        except Exception:
            # Si incluso esto falla, no hacemos nada más para no romper la tarea
            pass

        # Re-lanzamos la excepción para que se vea en los logs de Celery
        raise e
