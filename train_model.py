import pickle
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define industries and sample skills
industries = {
    "IT": ["Python", "Machine Learning", "Cloud Computing", "Cybersecurity", "Software Development"],
    "Healthcare": ["Patient Care", "Medical Coding", "Pharmaceuticals", "Clinical Research", "Health Informatics"],
    "Engineering": ["Mechanical Design", "Civil Engineering", "Electrical Systems", "CAD", "Project Management"],
    "Finance": ["Financial Analysis", "Risk Management", "Accounting", "Investment Strategies", "Auditing"],
    "Marketing": ["SEO", "Social Media Management", "Market Research", "Brand Strategy", "Content Creation"],
    "Education": ["Curriculum Design", "Teaching", "Educational Psychology", "Online Learning", "EdTech"],
    "Human Resources": ["Recruitment", "Talent Management", "HR Compliance", "Training & Development", "Payroll"],
    "Sales": ["Negotiation", "Lead Generation", "CRM", "Customer Relations", "Sales Strategy"]
}

# Generate synthetic resume data
resume_data = []
for industry, skills in industries.items():
    for _ in range(50):  # Generate 50 samples per industry
        resume_text = f"Experienced in {', '.join(random.sample(skills, 3))}. Strong background in {industry}."
        resume_data.append({"Resume": resume_text, "Industry": industry})

# Convert to DataFrame
df_resumes = pd.DataFrame(resume_data)

# Balance the dataset to avoid bias
min_samples = min(df_resumes['Industry'].value_counts())  # Find the lowest count
df_balanced = df_resumes.groupby('Industry').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_balanced["Resume"], df_balanced["Industry"], test_size=0.2, random_state=42)

# Train with a fresh TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Save the updated models
with open("updated_clf.pkl", "wb") as clf_file:
    pickle.dump(clf, clf_file)

with open("updated_tfidf.pkl", "wb") as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

print("✅ Model retraining completed. New classifier and TF-IDF vectorizer saved.")
