# /*
# *   VIRKER!
# */

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Læs datasættet ind til en DataFrame
df = pd.read_csv("data.csv")

# Split datasættet mellem features (X) og labels (y)
X = df["text"]
y = df["sentiment"]

# Konverter teksten til numeriske features ved brug af Tf-Idf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# clf modellen træner Random Forest model, som forudsiger
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Forudsiger fra datasættet ind til klassifikationsrapporten
y_pred = clf.predict(X)

# Print klassifikationsrapporten
print(classification_report(y, y_pred))


## USER INPUT

# Få user input
user_input = input("Analyser din tekst: ")

# Konverter teksten til numeriske features ved brug af Tf-Idf
user_input_features = vectorizer.transform([user_input])

# Forsudsig sentiment fra user input
predicted_sentiment = clf.predict(user_input_features)[0]

# Vælg og print sætningens grad (positiv, negativ, neutral)
if(predicted_sentiment == "Negative"):
    print("Din sætning er negativ. \n")
if(predicted_sentiment == "Positive"):
    print("Din sætning er positiv. \n")
if(predicted_sentiment == "Neutral"):
    print("Din sætning er neutral. \n")
# else:
#     print("Din sætning er", predicted_sentiment)


# Find vigtigheden af ordene
importance = clf.feature_importances_

# Find navnene
feature_names = vectorizer.get_feature_names_out()

# Laver en DataFrame til at beholde vigtigheden af ordene
importance_df = pd.DataFrame({'ord': feature_names, 'vigtighed': importance})

# Sorter data fra vigtighed
importance_df = importance_df.sort_values('vigtighed', ascending=False)

# Find top 5 vigtigste ord
top_5 = importance_df.head(5)

# Print de vigtigste ord
print("Analysen er baseret på følgende ord: \n" + "\n" + top_5.to_string(index=False))
