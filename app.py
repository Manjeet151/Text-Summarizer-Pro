from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
import PyPDF2
import docx
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='')
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Maximum word limit
MAX_WORDS = 1000  # Changed from 1500 to 1000

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

# Extract text from Word document
def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + " "
        return text
    except Exception as e:
        raise Exception(f"Error reading Word document: {str(e)}")

# Extract text from text file
def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading text file: {str(e)}")

# Text preprocessing
def preprocess_text(text):
    # Remove citation markers and special characters
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove anything in brackets
    text = re.sub(r'[^\w\s\.\,\;\?\!]', '', text)  # Keep only alphanumeric and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Limit text to maximum words
def limit_text_words(text, max_words=MAX_WORDS):
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

# Custom sentence tokenizer
def sentence_tokenize(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# Custom word tokenizer
def word_tokenize(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    return words

# Calculate sentence scores using TF-IDF
def calculate_tfidf_scores(sentences):
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    return sentence_scores

# Calculate sentence scores using TextRank algorithm
def calculate_textrank_scores(sentences):
    # Create similarity matrix
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Build graph and apply PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Convert to array in sentence order
    sentence_scores = np.array([scores[i] for i in range(len(sentences))])
    return sentence_scores

# Calculate sentence scores using position-based weighting
def calculate_position_scores(sentences):
    scores = []
    total_sentences = len(sentences)
    for i, sentence in enumerate(sentences):
        # First and last sentences get higher weights
        position_score = 1.0 - (abs(i - total_sentences/2) / total_sentences)
        scores.append(position_score)
    return np.array(scores)

# Calculate sentence scores using length-based weighting
def calculate_length_scores(sentences):
    scores = []
    word_counts = [len(word_tokenize(s)) for s in sentences]
    avg_length = np.mean(word_counts)
    
    for count in word_counts:
        # Sentences close to average length get higher scores
        length_score = 1.0 - (abs(count - avg_length) / max(word_counts))
        scores.append(length_score)
    
    return np.array(scores)

# Custom LSA-like algorithm
def calculate_lsa_scores(sentences):
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Perform SVD (Singular Value Decomposition)
    U, sigma, Vt = np.linalg.svd(tfidf_matrix.toarray(), full_matrices=False)
    
    # Use the first singular vector to score sentences
    sentence_scores = U[:, 0] ** 2
    return sentence_scores

# Main summarization function
def summarize_text(text, algorithm='hybrid', sentences_count=5):
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Limit text to maximum words
    limited_text = limit_text_words(cleaned_text)
    
    sentences = sentence_tokenize(limited_text)
    
    if len(sentences) <= sentences_count:
        return ' '.join(sentences)
    
    # Calculate scores based on selected algorithm
    if algorithm == 'tfidf':
        scores = calculate_tfidf_scores(sentences)
    elif algorithm == 'textrank':
        scores = calculate_textrank_scores(sentences)
    elif algorithm == 'position':
        scores = calculate_position_scores(sentences)
    elif algorithm == 'lsa':
        scores = calculate_lsa_scores(sentences)
    else:  # hybrid approach
        tfidf_scores = calculate_tfidf_scores(sentences)
        textrank_scores = calculate_textrank_scores(sentences)
        position_scores = calculate_position_scores(sentences)
        
        # Normalize and combine scores
        tfidf_scores = tfidf_scores / np.max(tfidf_scores)
        textrank_scores = textrank_scores / np.max(textrank_scores)
        position_scores = position_scores / np.max(position_scores)
        
        scores = 0.4 * tfidf_scores + 0.4 * textrank_scores + 0.2 * position_scores
    
    # Select top sentences
    top_sentence_indices = np.argsort(scores)[-sentences_count:]
    top_sentence_indices.sort()  # Maintain original order
    
    # Create summary
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    summary = ' '.join(summary_sentences)
    
    return summary

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        sentences_count = min(int(data.get('sentences_count', 5)), 20)  # Limit to 20 sentences
        algorithm = data.get('algorithm', 'hybrid')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Check if text exceeds word limit
        word_count = len(text.split())
        if word_count > MAX_WORDS:
            return jsonify({
                'error': f'Text exceeds maximum limit of {MAX_WORDS} words. Please reduce your text.',
                'word_count': word_count
            }), 400
        
        # Generate summary using our custom algorithm
        summary = summarize_text(text, algorithm, sentences_count)
        
        return jsonify({
            'summary': summary,
            'original_length': word_count,
            'summary_length': len(summary.split()),
            'algorithm_used': algorithm
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize-document', methods=['POST'])
def summarize_document():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text based on file type
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'pdf':
                text = extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                text = extract_text_from_docx(file_path)
            elif file_extension == 'txt':
                text = extract_text_from_txt(file_path)
            else:
                return jsonify({'error': 'Unsupported file type'}), 400
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            # Check if text exceeds word limit
            word_count = len(text.split())
            if word_count > MAX_WORDS:
                return jsonify({
                    'error': f'Document exceeds maximum limit of {MAX_WORDS} words. Please upload a shorter document.',
                    'word_count': word_count
                }), 400
            
            # Get parameters from form data
            sentences_count = min(int(request.form.get('sentences_count', 5)), 20)
            algorithm = request.form.get('algorithm', 'hybrid')
            
            # Generate summary using our custom algorithm
            summary = summarize_text(text, algorithm, sentences_count)
            
            return jsonify({
                'summary': summary,
                'original_length': word_count,
                'summary_length': len(summary.split()),
                'algorithm_used': algorithm
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)