from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Dummy function to build user-item matrix
def build_user_item_matrix(data):
    user_item_matrix = data.pivot_table(index='user_id', columns='product_id', values='purchase_amount')
    return user_item_matrix.fillna(0)

user_item_matrix = build_user_item_matrix(preprocessed_data)

# Calculating cosine similarity
user_similarity = cosine_similarity(user_item_matrix)
item_similarity = cosine_similarity(user_item_matrix.T)

def recommend_products(user_id, user_item_matrix, user_similarity, top_n=5):
    similar_users = np.argsort(-user_similarity[user_id])[:top_n]
    recommendations = []
    for similar_user in similar_users:
        user_recommendation_vector = user_item_matrix.iloc[similar_user]
        top_product_ids = user_recommendation_vector.argsort()[-top_n:][::-1]
        recommendations.extend(user_item_matrix.columns[top_product_ids])
    return set(recommendations)
