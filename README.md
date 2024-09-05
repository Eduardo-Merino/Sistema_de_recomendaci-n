# Sistema_de_recomendaci-n

Este proyecto construye un sistema de recomendación de películas utilizando técnicas de Machine Learning y despliega una API con FastAPI para servir recomendaciones de películas a los usuarios.

## Estructura del Proyecto

1. Exploración de Datos y Entrenamiento del Modelo (Prueba_técnica.ipynb)

El proceso comienza con la exploración de los datos de usuarios y películas, realizando diversos filtros para mejorar la calidad de los datos.
Se probaron varios modelos de recomendación. Finalmente, se seleccionó el modelo KNNBaseline de la librería Surprise, optimizado mediante GridSearch, utilizando el método de ALS para las estimaciones de base y el coeficiente de Pearson como métrica de similitud.
Después de entrenar el modelo y evaluar su rendimiento (RMSE: 0.8750), se desarrolló una función de recomendación basada en este modelo.
Los resultados del modelo y los datos se guardaron en archivos (modelo_recomendacion_peliculas.pkl y data_final.csv) para su posterior uso en la API.

2. Despliegue de la API (api_recomendacion_peliculas.py)

Se creó una API con FastAPI que carga el modelo de recomendación y los datos preprocesados.
La API expone un endpoint para obtener recomendaciones de películas dado el ID de un usuario.
La lógica principal de recomendación es proporcionada por la función recomendar_movies_para_usuario, que devuelve un diccionario con el número top de películas recomendadas, donde las claves son los títulos de las películas y los valores son diccionarios que contienen:
        - 'movieId': Identificador de la película.
        - 'rating_predicho': Calificación predicha por el modelo para el usuario.

## Cómo usar este proyecto

**Requisitos**
Python 3.12.5
Miniconda
Las librerías pueden ser instaladas mediante `conda env create -f environment.yml` , en el caso que tengas instalado conda (este fue el metodo yo use para instalar las dependecias del proyecto).

o con `pip install -r requirements.txt` (si ya tienes instalado python 3.12 y no tienes conda)

## Ejecución

1. Entrenamiento del modelo

Abre el archivo Prueba_técnica.ipynb en Jupyter Notebook.
Ejecuta las celdas para explorar los datos, entrenar el modelo y guardar los resultados. Cambia el path de u.data y u.item con el path donde se encuentran los datos en tu dispositivo

2. Despliegue de la API

Abre una terminal, activa tu enviroment y ejecuta el archivo api_recomendacion_peliculas.py con el siguiente comando:

`fastapi dev api_recomendacion_peliculas.py`

luego ingresa a http://127.0.0.1:8000/docs y has click en try out para usar el API Endpoint. Solamente tienes que selecionar un user_id para ejecutar el endpoint
![image](https://github.com/user-attachments/assets/5ded1e2a-9436-4f4d-9f44-e1d780467425)


