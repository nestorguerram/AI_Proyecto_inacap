# recommendation_system.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Datos de ejemplo: cada fila representa un usuario, cada columna un producto
# Los valores representan la calificación o interacción del usuario con el producto
user_product_matrix = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

def recommend_products(user_index, matrix, top_n=2):
    """
    Recomienda productos a un usuario basado en la similitud con otros usuarios.

    :param user_index: índice del usuario al que se le recomienda
    :param matrix: matriz de usuarios-productos
    :param top_n: número de productos a recomendar
    :return: índices de los productos recomendados
    """
    # Calcular similitud entre usuarios
    similarity = cosine_similarity(matrix)
    user_similarity = similarity[user_index]

    # Obtener puntuaciones ponderadas
    weighted_scores = user_similarity @ matrix
    user_data = matrix[user_index]

    # Evitar recomendar productos que ya ha visto
    unseen_indices = np.where(user_data == 0)[0]
    recommendations = [(i, weighted_scores[i]) for i in unseen_indices]

    # Ordenar por puntuación y devolver los mejores
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in recommendations[:top_n]]

if __name__ == "__main__":
    user = 0  # Usuario para el que queremos recomendaciones
    recommendations = recommend_products(user, user_product_matrix)
    print(f"Recomendaciones para el usuario {user}: {recommendations}")
