# Importar librerías necesarias
import json
import pandas as pd
import joblib
from fastapi import FastAPI, Response


app = FastAPI(
    title="Recomendación de Películas para Usuarios",
    description="API para la recomendación de películas para usuarios"
)

# Cargar el modelo
knn_baseline_model = joblib.load('modelo_recomendacion_peliculas.pkl')

# Cargar los datos
movies_user_data = pd.read_csv('data_final.csv')


def recomendar_movies_para_usuario(user_id: int, model, data: pd.DataFrame, num_recomendaciones: int = 10):
    """
    Genera una lista de recomendaciones de películas para un usuario en particular, basado en un modelo de predicción de calificaciones.

    Parámetros:
    ----------
    user_id : int
        ID del usuario para el cual se generan las recomendaciones.
    
    model : objeto del modelo de Surprise
        Modelo de predicción entrenado, que debe tener un método `predict(user_id, movie_id)` para predecir las calificaciones del usuario para películas específicas.
    
    data : pd.DataFrame
        DataFrame que contiene los datos de las interacciones entre usuarios y películas, con al menos las columnas:
        - 'userId': Identificador del usuario.
        - 'movieId': Identificador de la película.
        - 'title': Título de la película.
    
    num_recomendaciones : int, opcional, por defecto 10
        Número de películas a recomendar. El valor por defecto es 10.

    Retorna:
    -------
    dict
        Un diccionario con las top `num_recomendaciones` películas recomendadas, donde las claves son los títulos de las películas y los valores son diccionarios que contienen:
        - 'movieId': Identificador de la película.
        - 'rating_predicho': Calificación predicha por el modelo para el usuario.
    
    Funcionalidad:
    --------------
    1. Filtra las películas que el usuario ya ha visto.
    2. Identifica las películas que el usuario no ha visto.
    3. Predice la calificación para cada película no vista utilizando el modelo proporcionado.
    4. Ordena las películas por las calificaciones predichas en orden descendente.
    5. Retorna las `num_recomendaciones` películas con las calificaciones predichas más altas.
    """
    
    # Obtener todas las películas que el usuario ha visto
    user_data = data[data['userId'] == user_id]
    movies_vistas = set(user_data['movieId'].unique())
    
    # Obtener todas las películas disponibles en el dataset
    todas_las_movies = set(data['movieId'].unique())
    
    # Películas que el usuario no ha visto
    movies_no_vistas = todas_las_movies - movies_vistas
    
    # Crear un diccionario para almacenar las recomendaciones
    recomendaciones = {}
    
    # Para cada película no vista, predecir la calificación usando el modelo entrenado
    for movie_id in movies_no_vistas:
        pred = model.predict(user_id, movie_id)
        movie_title = data[data['movieId'] == movie_id]['title'].values[0]
        
        # Convertir el id de la película y la calificación predicha a tipos nativos de Python
        recomendaciones[movie_title] = {
            'movieId': int(movie_id),  # Asegurar que sea un tipo int nativo
            'rating_predicho': float(pred.est)  # Asegurar que sea un tipo float nativo
        }
    
    # Ordenar el diccionario según las calificaciones predichas, de mayor a menor
    recomendaciones = dict(sorted(recomendaciones.items(), key=lambda x: x[1]['rating_predicho'], reverse=True))
    
    # Seleccionar las top n películas
    top_recomendaciones = dict(list(recomendaciones.items())[:num_recomendaciones])
    
    # Retornar las películas recomendadas
    return top_recomendaciones


# Create the endpoint where the file is send it
@app.post("/recommend_movies", operation_id="recommend_movies_for_user")
def recommend_movies(
        user_id: int,
        model_path: str = None, 
        data_path: str = None,
        num_recomendaciones: int = 10,
    ):
    """
    Endpoint para generar recomendaciones de películas para un usuario específico.

    Este endpoint recibe el `user_id` de un usuario y, opcionalmente, las rutas a un modelo y a un conjunto de datos. 
    Genera un conjunto de recomendaciones de películas para el usuario utilizando un modelo de recomendación previamente entrenado. 

    Parámetros:
    ----------
    user_id : int
        El ID del usuario para el cual se generan las recomendaciones.
    
    model_path : str, opcional
        Ruta opcional para cargar un modelo de recomendación alternativo. Si no se proporciona, se utilizará el modelo previamente entrenado cargado al iniciar la API.
    
    data_path : str, opcional
        Ruta opcional para cargar un conjunto de datos alternativo con las interacciones entre usuarios y películas. Si no se proporciona, se utilizarán los datos predefinidos.
    
    num_recomendaciones : int, opcional, por defecto 10
        Número de recomendaciones a generar. El valor por defecto es 10.

    Retorna:
    --------
    Response
        Una respuesta JSON con el siguiente formato:
        - "status": Estado de la solicitud ("Éxito").
        - "mensaje": Mensaje de confirmación de que las recomendaciones fueron generadas correctamente.
        - "user_id": El ID del usuario para el que se generaron las recomendaciones.
        - "recomendaciones": Un diccionario con las películas recomendadas, donde las claves son los títulos de las películas y los valores son diccionarios que contienen:
            - 'movieId': El identificador de la película.
            - 'rating_predicho': La calificación predicha por el modelo para esa película.

    Funcionalidad:
    --------------
    1. Si se proporciona `data_path`, se carga un nuevo conjunto de datos desde el archivo especificado. Si no, se utilizan los datos predeterminados cargados inicialmente.
    2. Si se proporciona `model_path`, se carga un nuevo modelo de recomendación desde el archivo especificado. Si no, se utiliza el modelo predeterminado.
    3. Se llama a la función `recomendar_movies_para_usuario` para generar las recomendaciones.
    4. Se retorna una respuesta JSON con el estado, mensaje y recomendaciones generadas para el usuario.
    """

    # Comprobar si se cargaron nuevos datos si no usar el DataFrame usado en el Jupyter Notebook
    if data_path is not None:
        data = pd.read_csv(data_path)
    else:
        data = movies_user_data

    # Comprobar si se cargó un nuevo modelo si no usar el DataFrame usado en el Jupyter Notebook
    if model_path is not None:
        model = joblib.load(model_path)
    else:
        model = knn_baseline_model

    top_recomendaciones = recomendar_movies_para_usuario(
        user_id=user_id, 
        model=model, 
        data=data, 
        num_recomendaciones=num_recomendaciones
    )

    mensaje_final = {
                "status":"Éxito",
                "mensaje":"Las recomendaciones se generaron exitosamente",
                "user_id": user_id,
                "recomendaciones":top_recomendaciones,
            }
    
    return Response(
                content=json.dumps(mensaje_final),
                media_type="application/json;charset=UTF-8",
            )