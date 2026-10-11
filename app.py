from flask import Flask, request, jsonify
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon", quiet=True)

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    doc = nlp(text)

    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    score = sentiment_analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        sentiment = "positive"
    elif score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "sentiment_score": score,
        "entities": entities
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)