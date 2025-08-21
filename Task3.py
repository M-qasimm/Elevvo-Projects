# Task 4: Email/SMS Spam Classifier (using local dataset)

# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 2. Load dataset from local file
# Make sure your file path is correct (adjust if needed)
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

print("Sample data:")
print(df.head())

# 3. Split data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# 8. Test custom message
sample = ["Congratulations! You have won $1000 gift card"]
pred = model.predict(vectorizer.transform(sample))
print("Message:", sample[0])
print("Predicted:", pred[0])
