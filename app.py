from flask import Flask, request, render_template
import docx
import os
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# .docx file theke text pora
def read_docx(file):
    doc = docx.Document(file)
    return ' '.join([p.text for p in doc.paragraphs])

# text clean + tokenize
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return [w for w in words if w not in stop_words]

# keyword extract
def extract_keywords(text, top_n=10):
    words = clean_text(text)
    return Counter(words).most_common(top_n)

@app.route('/', methods=['GET', 'POST'])
def index():
    keywords = []
    similarity_matrix = []

    if request.method == 'POST':
        files = request.files.getlist('files')
        texts = [read_docx(f) for f in files]
        full_text = ' '.join(texts)

        # Keywords
        keywords = extract_keywords(full_text)

        # TF-IDF Similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

    return render_template('index.html', keywords=keywords, similarity=similarity_matrix)

if __name__ == '__main__':
    app.run(debug=True)
