import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0  # Fake
real["label"] = 1  # Real

data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)

data["content"] = data["title"] + " " + data["text"]

stopwords = set([
    'a', 'an', 'the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'by', 'it',
    'this', 'that', 'was', 'are', 'at', 'as', 'be', 'from', 'has', 'or', 'have', 'but',
    'not', 'you', 'they', 'we', 'he', 'she', 'them', 'his', 'her', 'their', 'its', 'if',
    'will', 'can', 'just', 'so', 'my', 'your'
])


def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    return ' '.join(word for word in words if word not in stopwords)

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]


vectorizer = TfidfVectorizer(max_df=0.7)
X_vect = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
