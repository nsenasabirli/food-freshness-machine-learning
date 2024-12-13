import pandas as pd
from rapidfuzz import fuzz


def map_flavors(dish_names, ingredients, flavor_keywords):
    predictions = []

    for dish, ingredient_list in zip(dish_names, ingredients):
        matched_flavors = set()

        for ingredient in ingredient_list.split(", "):
            for flavor in flavor_keywords:
                # Use fuzzy matching with a threshold (e.g., 80% similarity)
                if fuzz.partial_ratio(ingredient.lower(), flavor.lower()) > 80:
                    matched_flavors.add(flavor)

        if not matched_flavors:
            matched_flavors.add("unknown")

        predictions.append({
            "dish_name": dish,
            "predicted_flavors": ", ".join(matched_flavors)
        })

    return pd.DataFrame(predictions)
