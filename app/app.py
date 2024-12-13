from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from semantic_checker import semantic_similarity
from pdf_processor import expand_flavor_keywords  # Import the expansion function
import pandas as pd
import spacy

# Load the predicted flavors database
FLAVOR_DATABASE_PATH = '/Users/nsenasabirli/Downloads/PatternProject 2/TrainingTaste_Edibility.csv'

nlp = spacy.load("en_core_web_sm")

def extract_keywords(sentence):
    """
    Extract key terms (adjectives, nouns) from a sentence using spaCy.
    """
    doc = nlp(sentence)
    keywords = [token.text.lower() for token in doc if token.pos_ in {"ADJ", "NOUN"}]
    return keywords

def compare_flavors(user_sentence, predicted_flavor, expanded_flavors):
    """
    Compare user's input sentence with the predicted flavor using combined methods.
    """
    user_keywords = extract_keywords(user_sentence)
    best_score = 0.0

    for user_keyword in user_keywords:
        if user_keyword in expanded_flavors:
            return 1.0  # Perfect match

        fuzzy_score = fuzz.ratio(user_keyword.lower(), predicted_flavor.lower()) / 100
        semantic_score = semantic_similarity(user_keyword, predicted_flavor)
        best_score = max(best_score, fuzzy_score, semantic_score)

    return best_score

def determine_edibility(match_score):
    """
    Determine if the dish is edible based on the match score.
    """
    return "Edible" if match_score > 0.75 else "Potentially Spoiled"

def find_similar_dishes(dish_name, flavor_data):
    """
    Find similar dishes based on the dish name.
    """
    dish_names = flavor_data['dish_name'].str.lower().tolist()
    matches = process.extract(dish_name.lower(), dish_names, scorer=fuzz.ratio, limit=5)
    return [match[0] for match in matches if match[1] > 60]  # Return matches with a score > 60

def main():
    # Load the flavor database
    flavor_data = pd.read_csv(FLAVOR_DATABASE_PATH)

    # Prompt user for dish name
    dish_name = input("Enter the name of the dish: ").strip()

    # Search for similar dishes
    similar_dishes = find_similar_dishes(dish_name, flavor_data)

    if not similar_dishes:
        print(f"No dishes found similar to '{dish_name}'. Please try again.")
        return

    # If multiple similar dishes are found, ask the user to choose
    if len(similar_dishes) > 1:
        print("Which of the following dishes names is the closest to your dish?")
        for idx, dish in enumerate(similar_dishes, start=1):
            print(f"{idx}. {dish}")

        try:
            choice = int(input("Enter the number corresponding to your dish: "))
            selected_dish = similar_dishes[choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
            return
    else:
        selected_dish = similar_dishes[0]

    # Fetch dish data
    filtered_data = flavor_data[flavor_data['dish_name'].str.lower() == selected_dish.lower()]
    dish_data = filtered_data.iloc[0]
    predicted_flavors = dish_data["predicted_flavors"]

    # Expand the predicted flavors
    expanded_flavors = expand_flavor_keywords(predicted_flavors.split(", "))

    # Prompt the user for input
    user_input = input(f"What does the taste of {selected_dish} feel like? Describe it: ")

    # Compare the user's input with the predicted flavor
    scores = [
        compare_flavors(user_input, flavor.strip(), expanded_flavors)
        for flavor in predicted_flavors.split(", ")
    ]
    max_score = max(scores)  # Best match score
    best_flavor = predicted_flavors.split(", ")[scores.index(max_score)]

    # Determine edibility
    edibility = determine_edibility(max_score)

    # Display results
    print(f"\nDish Name: {selected_dish}")
    print(f"Your Flavor Input: {user_input}")
    print(f"Predicted Flavor: {best_flavor}")
    print(f"Match Score: {max_score:.2f}")
    print(f"Edibility: {edibility}")

if __name__ == "__main__":
    main()
