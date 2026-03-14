import os
import re
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder='templates')

# ================= CONFIGURATION =================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Predefined skill keywords for matching (lowercase)
SKILL_KEYWORDS = [
    'python', 'machine learning', 'tensorflow', 'pandas', 'numpy',
    'flask', 'django', 'html', 'css', 'javascript', 'git', 'sql',
    'pytorch', 'scikit-learn', 'keras', 'docker', 'aws', 'azure',
    'rest api', 'mongodb', 'postgresql', 'react', 'vue', 'node.js'
]

# Scoring weights
COSINE_WEIGHT = 70
SKILL_WEIGHT = 3

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SECRET_KEY'] = os.urandom(24)  # For session security

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD SEMANTIC MODEL =================
# Load Sentence Transformer model once at startup (global scope)
print("Loading semantic similarity model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load model: {e}")
    model = None  # Fallback handled in function


# ================= HELPER FUNCTIONS =================

def allowed_file(filename):
    """Check if the uploaded file has a valid PDF extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Removing special characters but preserving digits and common separators
    - Removing extra whitespace
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep letters, digits, spaces, +, ., #, /
    text = re.sub(r'[^a-z0-9\s\+\.\#\/]', ' ', text)
    
    # Remove extra whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.
    Returns cleaned text or empty string on error.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        
        # Return cleaned text
        return clean_text(text)
    
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""


def find_matched_skills(text, skill_list):
    """
    Find which predefined skills appear in the text.
    Returns a list of matched skills (not just count).
    """
    if not text:
        return []
    
    text_normalized = text.lower()
    matched = []
    
    for skill in skill_list:
        skill_normalized = skill.lower().replace('.', ' ').replace('-', ' ').replace('_', ' ')
        # Match with word boundaries, handling variations like node.js / NodeJS
        pattern = r'\b' + re.escape(skill_normalized).replace(r'\ ', r'[\s\.\-\_]+') + r'\b'
        if re.search(pattern, text_normalized, re.IGNORECASE):
            matched.append(skill)
    
    return matched


def calculate_similarity(job_description, resume_texts):
    """
    Calculates semantic cosine similarity between job description and resumes 
    using Sentence Transformers embeddings.
    Returns a list of similarity scores (0.0 to 1.0).
    """
    global model
    
    # Filter out empty texts and track original indices
    valid_data = [(i, txt) for i, txt in enumerate(resume_texts) if txt and txt.strip()]
    
    if not valid_data or model is None:
        # Fallback: return zeros if no valid data or model failed to load
        return [0.0] * len(resume_texts)
    
    try:
        # Prepare texts for encoding
        texts_to_encode = [job_description] + [txt for _, txt in valid_data]
        
        # Generate embeddings
        embeddings = model.encode(texts_to_encode, convert_to_tensor=False, show_progress_bar=False)
        
        # Calculate cosine similarity: Job Desc (index 0) vs all Resumes (index 1:)
        job_embedding = embeddings[0:1]
        resume_embeddings = embeddings[1:]
        
        cosine_sim = cosine_similarity(job_embedding, resume_embeddings).flatten()
        
        # Map scores back to original indices
        full_scores = [0.0] * len(resume_texts)
        for idx, (orig_idx, _) in enumerate(valid_data):
            full_scores[orig_idx] = float(cosine_sim[idx])
        
        return full_scores
        
    except Exception as e:
        print(f"Similarity calculation error: {str(e)}")
        return [0.0] * len(resume_texts)


def calculate_final_score(cosine_sim, skill_count):
    """
    Combine cosine similarity and skill matches into final score.
    Formula: (Cosine Similarity × 70) + (Skill Count × 3)
    Returns score rounded to 2 decimal places, capped at 100.
    """
    # Cosine similarity is 0-1, multiply by 70 to get 0-70 range
    cosine_component = cosine_sim * COSINE_WEIGHT
    
    # Skill count multiplied by weight
    skill_component = skill_count * SKILL_WEIGHT
    
    # Final score capped at 100
    final_score = min(cosine_component + skill_component, 100.0)
    
    return round(final_score, 2)


def rank_resumes(job_description, resume_files):
    """
    Main logic function to process uploaded files and rank resumes.
    Returns a list of dictionaries with filename, scores, skills, and details.
    """
    resume_data = []  # List of dicts: {filename, text, skills, skill_count}
    
    # ================= STEP 1: Extract text from all uploaded PDFs =================
    for file in resume_files:
        if not file or not allowed_file(file.filename):
            continue
        
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save file temporarily
            file.save(filepath)
            
            # Extract and clean text
            text = extract_text_from_pdf(filepath)
            
            # Error handling: Skip if no text extracted
            if not text or len(text.strip()) < 50:  # Minimum 50 chars to be valid
                print(f"Warning: Skipping {filename} - insufficient text extracted")
                continue
            
            # Find matched skills (returns list)
            matched_skills = find_matched_skills(text, SKILL_KEYWORDS)
            
            resume_data.append({
                'filename': filename,
                'text': text,
                'skills': matched_skills,
                'skill_count': len(matched_skills)
            })
            
            # Clean up: Remove temporary file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # If no resumes were successfully processed
    if not resume_data:
        return []
    # ================= STEP 2: Calculate Semantic Similarity =================
    resume_texts = [data['text'] for data in resume_data]
    cleaned_job_desc = clean_text(job_description)
    
    cosine_scores = calculate_similarity(cleaned_job_desc, resume_texts)
    
    # ================= STEP 3: Calculate Final Scores =================
    results = []
    for i, data in enumerate(resume_data):
        cosine_sim = cosine_scores[i] if i < len(cosine_scores) else 0.0
        
        final_score = calculate_final_score(
            cosine_sim=cosine_sim,
            skill_count=data['skill_count']
        )
        
        results.append({
            'filename': data['filename'],
            'cosine_score': round(cosine_sim * 100, 2),  # For display (percentage)
            'skill_matches': data['skill_count'],
            'skills': data['skills'],  # ✅ NEW: List of matched skills
            'final_score': final_score
        })
    
    # ================= STEP 4: Sort by Final Score (Descending) =================
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return results


def truncate_text(text, max_length=100, suffix="..."):
    """
    Safely truncate text with ellipsis.
    Handles None and short strings safely.
    """
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


# ================= FLASK ROUTES =================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route: Handle form display and resume screening submission.
    """
    if request.method == 'POST':
        # Get form data with safe defaults
        job_description = (request.form.get('job_description', '') or '').strip()
        resume_files = request.files.getlist('resumes')
        
        # Validation
        if not job_description or len(job_description) < 20:
            return render_template('index.html', error="Please enter a valid job description (minimum 20 characters).")
        
        if not resume_files or all(f.filename == '' for f in resume_files):
            return render_template('index.html', error="Please upload at least one resume PDF.")
        
        # Process and rank resumes
        ranked_results = rank_resumes(job_description, resume_files)
        
        # Handle case where no resumes were successfully processed
        if not ranked_results:
            return render_template('index.html', error="No valid resumes could be processed. Please check your PDF files.")
        
        # ✅ STEP 4: Identify top candidate
        best_resume = ranked_results[0]['filename']
        
        # Safely truncate job description for display
        job_desc_preview = truncate_text(job_description, 100)
        
        # Display results with all required fields
        return render_template('results.html', 
                             results=ranked_results, 
                             job_desc=job_desc_preview,
                             full_job_desc=job_description,
                             best_resume=best_resume)  # ✅ Pass top candidate
    
    # GET request: Show upload form
    return render_template(
    'index.html',
    error=None,
    skill_keywords=SKILL_KEYWORDS
)


@app.route('/health')
def health_check():
    """Simple health check endpoint for testing."""
    return {'status': 'ok', 'message': 'AI Resume Analyzer is running'}


# ================= MAIN ENTRY POINT =================

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 AI Resume Analyzer - Advanced Semantic Screening")
    print("=" * 60)
    print(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🎯 Loaded {len(SKILL_KEYWORDS)} skill keywords for matching")
    print(f"🧮 Scoring: (Cosine×{COSINE_WEIGHT}) + (Skills×{SKILL_WEIGHT}), capped at 100")
    print(f"🔍 Model: all-MiniLM-L6-v2 (semantic embeddings)" if model else "⚠️ Model: Fallback mode (no embeddings)")
    print("-" * 60)
    print("🚀 Starting Flask server in debug mode...")
    print("🌐 Access at: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)