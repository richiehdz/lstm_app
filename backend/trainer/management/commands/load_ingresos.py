from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
import pandas as pd

from trainer.models import IngresoSemestral


class Command(BaseCommand):
    help = "Carga el archivo cant_alumnos_ingreso_por_semestre.xlsx en la tabla IngresoSemestral"

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            type=str,
            default="data/cant_alumnos_ingreso_por_semestre.xlsx",
            help="Ruta al archivo de Excel relativa al BASE_DIR",
        )

    def handle(self, *args, **options):
        rel_path = options["path"]
        file_path = Path(settings.BASE_DIR) / rel_path

        if not file_path.exists():
            self.stderr.write(self.style.ERROR(f"No se encontró el archivo: {file_path}"))
            return

        self.stdout.write(f"Leyendo: {file_path}")
        df = pd.read_excel(file_path)

        # Esperamos columnas 'Semestre' y 'Ingresados'
        if not {"Semestre", "Ingresados"} <= set(df.columns):
            self.stderr.write(self.style.ERROR("El archivo debe tener columnas 'Semestre' e 'Ingresados'"))
            return

        df["Semestre"] = df["Semestre"].astype(str).str.strip().str.upper()
        df["Ingresados"] = pd.to_numeric(df["Ingresados"], errors="coerce").fillna(0).astype(int)

        creados = 0
        actualizados = 0

        for _, row in df.iterrows():
            sem = row["Semestre"]
            ing = int(row["Ingresados"])

            obj, created = IngresoSemestral.objects.update_or_create(
                semestre=sem,
                defaults={"ingresados": ing},
            )
            if created:
                creados += 1
            else:
                actualizados += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Proceso terminado. Creados: {creados}, Actualizados: {actualizados}"
            )
        )
