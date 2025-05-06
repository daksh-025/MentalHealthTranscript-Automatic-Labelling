import os
import PyPDF2
import spacy
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model for preprocessing
nlp = spacy.load('en_core_web_sm')

# Predefined mental health labels and their associated keywords
mental_health_labels = {
    'Depression': ['sad', 'hopeless', 'fatigue', 'loss', 'interest', 'sleep', 'unmotivated'],
    'Anxiety': ['fear', 'worry', 'restlessness', 'irritability', 'tension', 'nervous'],
    'PTSD': ['trauma', 'flashbacks', 'nightmares', 'avoidance', 'fear', 'distress'],
    'OCD': ['obsession', 'compulsion', 'ritual', 'control', 'intrusive', 'repetition'],
    'ADHD': ['focus', 'restlessness', 'impulsive', 'disorganized', 'hyperactivity', 'attention']
}

# Step 1: Extract text from PDF files
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Preprocess text using spaCy (lemmatization, remove stopwords)
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens

# Step 3: Train Word2Vec model on the transcripts (or use pre-trained Word2Vec)
def train_word2vec(corpus):
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Step 4: Get the average vector of a document (or label)
def get_average_vector(text_tokens, word2vec_model):
    vector = np.zeros(word2vec_model.vector_size)
    count = 0
    for word in text_tokens:
        if word in word2vec_model.wv:
            vector += word2vec_model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    return vector

# Step 5: Calculate cosine similarity between transcript and label vectors
def calculate_similarity(transcript_vector, label_vector):
    return cosine_similarity([transcript_vector], [label_vector])[0][0]

# Step 6: Match each transcript to the closest mental health label
def match_transcripts_to_labels(transcripts, word2vec_model):
    label_vectors = {}
    
    # Generate average vector for each mental health label based on keywords
    for label, keywords in mental_health_labels.items():
        label_vectors[label] = get_average_vector(keywords, word2vec_model)
    
    # Match each transcript to the label with the highest cosine similarity
    for transcript in transcripts:
        transcript_vector = get_average_vector(transcript, word2vec_model)
        similarities = {label: calculate_similarity(transcript_vector, label_vector) for label, label_vector in label_vectors.items()}
        best_match = max(similarities, key=similarities.get)
        print(f"Transcript matches with {best_match} (Similarity: {similarities[best_match]:.4f})")

# Main function to process PDF transcripts and match them to mental health labels
def main():
    transcripts_folder = "Transcripts"  # Folder containing the PDFs
    corpus = []
    transcript_texts = []
    
    # Step 1: Extract and preprocess transcripts
    for pdf_file in os.listdir(transcripts_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(transcripts_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            preprocessed_tokens = preprocess_text(text)
            transcript_texts.append(preprocessed_tokens)
            corpus.append(preprocessed_tokens)
    
    # Step 2: Train Word2Vec model on the corpus of transcripts
    word2vec_model = train_word2vec(corpus)
    
    # Step 3: Match transcripts to predefined mental health labels using cosine similarity
    match_transcripts_to_labels(transcript_texts, word2vec_model)

if __name__ == "__main__":
    main()
