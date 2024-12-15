import os
import dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

def generate_descriptions(folder_path, model):
    descriptions = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(folder_path, file_name)
            with open(image_path, "rb"):
                response = model.generate_content(image_path)
                descriptions[file_name] = response.text
    return descriptions

def search_memes(descriptions, query):
    file_names = list(descriptions.keys())
    text_data = list(descriptions.values())

    text_data.append(query)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)

    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()

    top_indices = similarity_scores.argsort()[-5:][::-1]
    top_files = [file_names[i] for i in top_indices]

    return top_files

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    folder_path = "memes"
    query = input("Enter your search query: ")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    descriptions = generate_descriptions(folder_path, model)

    top_memes = search_memes(descriptions, query)
    print("Top 5 memes related to your query:")
    for meme in top_memes:
        print(meme)

if __name__ == "__main__":
    main()
