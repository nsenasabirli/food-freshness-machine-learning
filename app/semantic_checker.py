from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast


def semantic_similarity(user_input, predicted_flavor):
    """
    Calculate semantic similarity between user input and predicted flavor.
    """
    user_embedding = model.encode(user_input.lower(), convert_to_tensor=True)
    flavor_embedding = model.encode(predicted_flavor.lower(), convert_to_tensor=True)

    # Use cosine similarity
    similarity = util.cos_sim(user_embedding, flavor_embedding).item()
    return similarity
