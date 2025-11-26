import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")

# Carga config desde settings.py usando el namespace CELERY_
app.config_from_object("django.conf:settings", namespace="CELERY")

# Descubre tasks.py en todas las apps instaladas
app.autodiscover_tasks()
