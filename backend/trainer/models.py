from django.db import models
import uuid


class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    n_rows = models.PositiveIntegerField(null=True, blank=True)
    n_cols = models.PositiveIntegerField(null=True, blank=True)
    columns_schema = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.original_name


class TrainingJob(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pendiente"
        RUNNING = "RUNNING", "En ejecución"
        COMPLETED = "COMPLETED", "Completado"
        FAILED = "FAILED", "Fallido"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="jobs")
    output_path = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    progress = models.PositiveIntegerField(default=0)

    target_column = models.CharField(max_length=255)
    input_columns = models.JSONField()

    window_size = models.PositiveIntegerField(default=10)
    epochs = models.PositiveIntegerField(default=10)

    loss_final = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)

    model_path = models.CharField(
        max_length=500,
        null=True,
        blank=True,
        help_text="Ruta en disco del modelo entrenado",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Job {self.id} ({self.status})"
class IngresoSemestral(models.Model):
    semestre = models.CharField(
        max_length=6,
        unique=True,
        help_text="Clave de semestre, por ejemplo 2025A, 2024B, etc.",
    )
    ingresados = models.PositiveIntegerField(
        help_text="Número de alumnos de nuevo ingreso en ese semestre",
    )

    def __str__(self):
        return f"{self.semestre}: {self.ingresados}"