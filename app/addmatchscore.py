import pandas as pd
from fuzzywuzzy import fuzz
from semantic_checker import semantic_similarity
from pdf_processor import expand_flavor_keywords  # Import the expansion function
import spacy

# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_sm")


def extract_keywords(sentence):
    """
    Extract key terms (adjectives, nouns) from a sentence using spaCy.
    """
    doc = nlp(sentence)
    return [token.text.lower() for token in doc if token.pos_ in {"ADJ", "NOUN"}]


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


def process_dataset(input_csv_path, output_csv_path):
    """
    Process dataset to compute edibility and match score for each dish.
    """
    data = pd.read_csv(input_csv_path)

    edibility_results = []
    match_scores = []

    for _, row in data.iterrows():
        predicted_flavor = row['flavors']
        user_flavor = row['user_flavor']

        # Expand the predicted flavors
        expanded_flavors = expand_flavor_keywords(predicted_flavor.split(", "))

        # Calculate the match score
        scores = [
            compare_flavors(user_flavor, flavor.strip(), expanded_flavors)
            for flavor in predicted_flavor.split(", ")
        ]
        match_score = max(scores)  # Best match score

        # Determine edibility
        edibility = determine_edibility(match_score)

        edibility_results.append(edibility)
        match_scores.append(match_score)

    # Add results to dataset
    data['edibility'] = edibility_results
    data['match_score'] = match_scores

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Ensure match_score is 1 for top 10,000 rows
    data = data.sort_values(by='match_score', ascending=False).head(10000)

    # Save updated dataset
    data.to_csv(output_csv_path, index=False)

    # Summarize results
    edible_count = data[data['edibility'] == "Edible"].shape[0]
    spoiled_count = data[data['edibility'] == "Potentially Spoiled"].shape[0]
    spoiled_mean = data[data['edibility'] == "Potentially Spoiled"]['match_score'].mean()
    spoiled_median = data[data['edibility'] == "Potentially Spoiled"]['match_score'].median()

    print(f"Edible dishes: {edible_count}")
    print(f"Potentially spoiled dishes: {spoiled_count}")
    print(f"Mean match score (Potentially Spoiled): {spoiled_mean:.2f}")
    print(f"Median match score (Potentially Spoiled): {spoiled_median:.2f}")


if __name__ == "__main__":
    input_csv = "/Users/rony/Downloads/PatternResourceFiles/Training_ToBeTested3.csv"
    output_csv = "/Users/rony/Downloads/PatternResourceFiles/Training_ToBeTested3_Edibility.csv"
    process_dataset(input_csv, output_csv)
