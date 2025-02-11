from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


corpus = "When we were in Paris we visited a lot of museums. We first went to the Louvre, the largest art museum in the world. I have always been interested in art so I spent many hours there. The museum is enourmous, so a week there would not be enough."


sentences = sent_tokenize(corpus)
words = word_tokenize(corpus)
unique_stops = set(stopwords.words("english"))
no_stops = [
    word for word in words if word.lower() not in unique_stops and word.isalpha()
]


lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
lancaster = LancasterStemmer()


lemmatized = [lemmatizer.lemmatize(word) for word in no_stops]
# stemmed = [porter.stem(word) for word in no_stops]

bow_counter = Counter(lemmatized)

vectorizer = CountVectorizer(
    analyzer="word",
    stop_words="english",
    max_features=500,
    lowercase=True,
)

vectorized = vectorizer.fit_transform(sentences)
tokens = vectorizer.get_feature_names_out()
as_array = vectorized.toarray()
vocab = vectorizer.vocabulary_

index_list = []

for i, _ in enumerate(vectorized):
    index_list.append(f"Sentence_{i}")

dataframe = pd.DataFrame(as_array, index=index_list, columns=tokens)

tfid_transformer_model = TfidfTransformer(
    # analyzer="word",
    # stop_words="english",
    # max_features=500,
    # lowercase=True,
    smooth_idf=True,
    use_idf=True,
)

tfid_transformer_model.fit(vectorized)
tidf_vector = tfid_transformer_model.transform(vectorized)

transformed_data = pd.DataFrame(
    tfid_transformer_model.idf_, index=tokens, columns=["idf_weights"]
)

vectorized_tfidf_df = pd.DataFrame(
    tidf_vector.toarray(), columns=vectorizer.get_feature_names_out()
)
# ==================================================

tf_vectorizer_model = TfidfVectorizer(
    analyzer="word",
    stop_words="english",
    max_features=500,
    lowercase=True,
    use_idf=True,
)

vectorized = tf_vectorizer_model.fit_transform(sentences)
feature_names = tf_vectorizer_model.get_feature_names_out()

vectorized_df = pd.DataFrame(
    tf_vectorizer_model.idf_, index=feature_names, columns=["idf_weights"]
)
tdf_df = pd.DataFrame(vectorized.toarray(), columns=feature_names)
mean_weights = np.asarray(vectorized.mean(axis=0)).ravel().tolist()

mean_df = pd.DataFrame({"mean_weights": mean_weights}, index=feature_names)
# ========================================

text1 = "all the world’s a stage, and all the men and women merely players"
text2 = "you must be the change you wish to see in the world"
corpus = [text1, text2]
vectorizer = TfidfVectorizer(stop_words="english", use_idf=True)
vectorized = vectorizer.fit_transform(corpus)

vector1 = vectorized.toarray()[0]
vector2 = vectorized.toarray()[1]


vector1Len = np.linalg.norm(vector1)
vector2Len = np.linalg.norm(vector2)

cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector1))
angle_radians = np.arccos(cosine)
angle_degrees = angle_radians * 360 / 2 / np.pi
round(angle_degrees, 2)

print(angle_radians)
# Cluster =============================================
text = """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data.
It applies knowledge and actionable insights from data across a broad range of application domains.
Data science is related to data mining, machine learning and big data.
The Bolshoi Theatre is a historic theatre in Moscow, Russia.
It was originally designed by architect Joseph Bové, which holds ballet and opera performances.
Before the October Revolution it was a part of the Imperial Theatres of the Russian Empire along with Maly Theatre in Moscow and a few theatres in Saint Petersburg.
Data science is a concept to unify statistics, data analysis, informatics, and their related methods in order to understand and analyze actual phenomena with data.
However, data science is different from computer science and information science.
The main building of the theatre, rebuilt and renovated several times during its history, is a landmark of Moscow and Russia.
On 28 October 2011, the Bolshoi re-opened after an extensive six-year renovation.
"""
corpus = []
for line in text.split("\n"):
    if line:
        line = line.lower()
        corpus.append(line)

tfIdfVectorizer = TfidfVectorizer(use_idf=True, stop_words="english")
vectorized = tfIdfVectorizer.fit_transform(corpus)

kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=42)
fitted = kmeans_model.fit(vectorized)

prediction = [
    "The Bolshoi Theatre is a historic theatre in Moscow, Russia.",
    "Space is  big.",
    "Many statisticians, including Nate Silver, have argued that data science is not a new field, but rather another name for statistics.",
    "Urusov set up the theatre in collaboration with English tightrope walker Michael Maddox.",
    "Until the mid-1990s, most foreign operas were sung in Russian, but Italian and other languages have been heard more frequently on the Bolshoi stage in recent years.",
]

predicted = kmeans_model.predict(tfIdfVectorizer.transform(prediction))

print(predicted)
