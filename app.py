from flask import Flask, request, jsonify, render_template
from deep_translator import GoogleTranslator  # Use deep_translator instead
import joblib

app = Flask(__name__, template_folder="templates")

# Function to translate text to English
def translate_to_english(text: str, src_language: str) -> str:
    try:
        translator = GoogleTranslator(source=src_language, target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return "Translation unavailable"

# Language codes for translation
language_codes = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Russian": "ru",
    "Swedish": "sv",
    "Malayalam": "ml",
    "Dutch": "nl",
    "Arabic": "ar",
    "Turkish": "tr",
    "German": "de",
    "Tamil": "ta",
    "Danish": "da",
    "Kannada": "kn",
    "Greek": "el",
    "Hindi": "hi"
}

# Load pre-trained models
try:
    model = joblib.load("language_detection_model.pkl")
    le = joblib.load("label_encoder.pkl")
    cv = joblib.load("count_vectorizer.pkl")
except Exception as e:
    print(f"Model Loading Error: {e}")
    model, le, cv = None, None, None

# Function to predict language and translate
def predict(text):
    if not model or not le or not cv:
        return "Model not loaded", "Translation unavailable"
    
    x = cv.transform([text])  # Convert text to bag-of-words
    lang_index = model.predict(x)  # Predict language index
    lang_name = le.inverse_transform(lang_index)[0]  # Convert index to language name
    
    # Get language code (fallback to 'auto' if not found)
    src_code = language_codes.get(lang_name, 'auto')
    
    # Translate only if it's not already English
    translation = text if lang_name == "English" else translate_to_english(text, src_code)
    
    return lang_name, translation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-language', methods=['POST'])
def detect_language():
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        language, translation = predict(text)
        return jsonify({"language": language, "translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
