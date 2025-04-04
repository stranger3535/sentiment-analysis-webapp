
# Create your views here.
from django.shortcuts import render
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def home(request):
    sentiment = ""
    if request.method == "POST":
        text = request.POST.get("text", "")
        if text:
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

    return render(request, "index.html", {"sentiment": sentiment})