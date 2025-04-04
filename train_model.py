import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset (add more samples for better accuracy)
data = {
    "text": [
        "I love this product!",
        "This is the worst!",
        "I am happy with my purchase",
        "Terrible experience",
        "Amazing quality!",
        "Very bad service",
        "Absolutely fantastic",
        "I hate it",
        "This is awesome",
        "Not good at all"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training complete and saved!")