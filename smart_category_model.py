import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load data
df = pd.read_csv('expense_data.csv')

# Create model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train
model.fit(df['description'], df['category'])

# Save model
joblib.dump(model, 'category_model.pkl')
print("✅ Model trained and saved as category_model.pkl")
