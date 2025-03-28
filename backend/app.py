from flask import Flask, request, jsonify #type: ignore
import joblib #type: ignore

# Carregar modelo e vetorizar
model = joblib.load("../sentiment_model.pkl")
vectorizer = joblib.load("../vectorizer.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "avaliacao" not in data:
        return jsonify({"error": "Avaliação não fornecida"}), 400

    # Transformar a avaliação para o formato do modelo
    text_vectorized = vectorizer.transform([data["avaliacao"]])
    prediction = model.predict(text_vectorized)[0]

    return jsonify({"sentimento": prediction})

if __name__ == "__main__":
    app.run(debug=True)
