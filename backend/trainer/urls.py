from django.urls import path
from . import views

app_name = "trainer"

urlpatterns = [
    path("", views.home, name="home"),
    path("upload_and_train/", views.upload_and_train, name="upload_and_train"),
    path("jobs/<uuid:job_id>/", views.job_detail, name="job_detail"),
    path(
        "resumir-datasets-originales/",
        views.resumir_datasets_originales_view,
        name="resumir_datasets_originales",
    ),
    path(
    "unir-datasets-resumidos/",
    views.unir_datasets_resumidos_view,
    name="unir_datasets_resumidos",
),
    path(
        "combinar-preregistro/",
        views.combinar_oferta_y_preregistro_view,
        name="combinar_preregistro",
    ),
        path(
        "training-jobs/<uuid:pk>/",
        views.TrainingJobDetailView.as_view(),
        name="trainingjob-detail",
    ),
    path(
        "training-jobs/<uuid:pk>/download/",
        views.TrainingJobDownloadView.as_view(),
        name="trainingjob-download",
    ),
    path(
        "comparar-predicciones/",
        views.comparar_predicciones_view,
        name="comparar_predicciones",
    ),
    path(
        "comparar-predicciones/descargar/",
        views.descargar_comparacion_csv,
        name="descargar_comparacion_csv",
    ),
]
