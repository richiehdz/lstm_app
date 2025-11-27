from django import forms


class UploadDatasetForm(forms.Form):
    file = forms.FileField(label="Dataset CSV")
    target_column = forms.CharField(label="Columna objetivo", max_length=255)
    input_columns = forms.CharField(
        label="Columnas de entrada",
        help_text="Nombres de columnas separados por comas",
    )
    window_size = forms.IntegerField(label="Tamaño de ventana", initial=10, min_value=1)
    epochs = forms.IntegerField(label="Épocas", initial=5, min_value=1)


# 👇 Form "vacío" solo para tener csrf y validación básica
class MultiCSVUploadForm(forms.Form):
    # No definimos FileField aquí, los archivos los leeremos desde request.FILES
    pass


class CompararPrediccionesForm(forms.Form):
    predicciones_file = forms.FileField(
        label="Archivo de predicciones (CSV)",
        help_text="Por ejemplo: job_xxx_predicciones.csv"
    )
    reales_file = forms.FileField(
        label="Archivo de datos reales (CSV)",
        help_text="Calendario futuro con valores reales"
    )