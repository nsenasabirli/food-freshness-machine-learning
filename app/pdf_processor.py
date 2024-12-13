import PyPDF2
import os
from nltk.corpus import wordnet
import nltk

# Ensure required resources are downloaded
nltk.download('wordnet')

def get_synonyms(word):
    """
    Get synonyms for a word from WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def expand_flavor_keywords(flavor_keywords):
    """
    Expand flavor keywords with their synonyms.
    """
    expanded_keywords = set(flavor_keywords)
    for flavor in flavor_keywords:
        expanded_keywords.update(get_synonyms(flavor))
    return list(expanded_keywords)

def extract_flavors(articles_dir):
    """
    Extract flavor-related keywords from PDFs.
    """
    flavor_keywords = set()  # Use a set to avoid duplicates
    common_flavors = [
        "sweet", "sugary", "caramelized", "honeyed",
        "spicy", "hot", "pungent", "peppery",
        "savory", "umami", "meaty", "brothy",
        "sour", "tangy", "tart", "acidic",
        "bitter", "astringent", "sharp",
        "salty", "briny", "cured",
        "fruity", "citrusy", "herbal", "earthy",
        "nutty", "smoky", "creamy", "buttery"
    ]

    # Expand common flavors with synonyms
    expanded_flavors = expand_flavor_keywords(common_flavors)

    for pdf_file in os.listdir(articles_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(articles_dir, pdf_file)
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    for flavor in expanded_flavors:
                        if flavor in text.lower():
                            flavor_keywords.add(flavor)
    return list(flavor_keywords)