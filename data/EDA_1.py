import pandas as pd
import ast

df = pd.read_csv(r"D:\USTH\nlp\NLP_Sentiment_analysis_and_classification\data\test.csv")

df["label_name"] = df["label_name"].apply(ast.literal_eval)

# Count the number of labels for each emotion

emotion_counts = df["label_name"].explode().value_counts()

print(emotion_counts)

# Count the number of emotions per sample

print("\nNumber of emotions per sample:")
df["num_emotions"] = df["label_name"].apply(len)

emotion_per_sample = df["num_emotions"].value_counts().sort_index()

print(emotion_per_sample)

# Find the correlation between emotions
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

emotion_matrix = pd.DataFrame(
    mlb.fit_transform(df["label_name"]),
    columns=mlb.classes_
)

corr = emotion_matrix.corr()
co_matrix = emotion_matrix.T.dot(emotion_matrix)
pairs = []

emotions = corr.columns

for i in range(len(emotions)):
    for j in range(i+1, len(emotions)):
        
        e1 = emotions[i]
        e2 = emotions[j]
        
        correlation = corr.loc[e1, e2]
        co_occurrence = co_matrix.loc[e1, e2]
        
        pairs.append((e1, e2, correlation, co_occurrence))
        
pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

top20 = pairs_sorted[:20]
print("Top 20 emotion pairs by correlation:")

for e1, e2, corr_val, count in top20:
    print(f"{e1:15} {e2:15} corr={corr_val:.3f}  co-occur={count}")
    
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm")

# Sentence length distribution
df["char_len"] = df["text"].str.len()
df["word_len"] = df["text"].str.split().apply(len)
print("Character length statistics:")
print(df["char_len"].describe())

print("\nWord length statistics:")
print(df["word_len"].describe())

longest = df.sort_values("word_len", ascending=False).head(10)

print(longest[["word_len","text"]])

shortest = df.sort_values("word_len").head(10)

print(shortest[["word_len","text"]])

emotion_matrix["length"] = df["word_len"]

emotion_length = emotion_matrix.groupby("length").mean()

print("\nAverage emotion presence by sentence length:")
print(emotion_length.head(10))

plt.figure(figsize=(12,10))
plt.hist(df["word_len"], bins=50)
plt.xlabel("Sentence length (words)")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")

plt.show()