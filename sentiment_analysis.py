import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 📥 Load dataset from CSV file
df = pd.read_csv("sentiment_dataset.csv")

# 🧹 Optional: Clean data (drop empty rows)
df.dropna(inplace=True)

# Split into features and labels
X = df['text']
y = df['sentiment']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model using TF-IDF
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print()
while True:
    your_input = input("Enter a sentence to analyze sentiment (or type 'exit' to quit): ")
    if your_input.lower() == 'exit':
        print("Goodbye! 👋")
        break
    prediction = model.predict([your_input])[0]
    print("Predicted Sentiment:", prediction)
    print()  # Just a line space for readability
