from celery import shared_task
from django.utils import timezone
from time import sleep

from .models import TrainingJob


@shared_task
def train_lstm_job(job_id: str):
    job = TrainingJob.objects.get(id=job_id)

    job.status = TrainingJob.Status.RUNNING
    job.started_at = timezone.now()
    job.progress = 0
    job.save()

    # Simulación de entrenamiento: 10 pasos
    steps = 10
    for i in range(steps):
        sleep(1)  # aquí luego irá el entrenamiento real
        job.progress = int((i + 1) / steps * 100)
        job.save(update_fields=["progress"])

    # Resultado dummy
    job.loss_final = 0.1234
    job.status = TrainingJob.Status.COMPLETED
    job.finished_at = timezone.now()
    job.model_path = f"models/{job.id}.bin"
    job.save()
