import pandas as pd

def load_ingredients(dataset_path):
    """
    Load the ingredients and dish names from the dataset.
    """
    data = pd.read_csv(dataset_path)
    # Assuming the dataset has columns 'ingredients' and 'dish_name'
    dish_names = data['Title'].dropna().tolist()
    ingredients = data['Ingredients'].dropna().tolist()
    return ingredients, dish_names
