# src/clinical_nlp.py

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def extract_keywords(text_series, top_n=50):
    """
    Extract top N keywords from a pandas Series of text
    """
    vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english')
    vectorizer.fit(text_series)
    return vectorizer.get_feature_names_out()

def generate_wordcloud(text_series, width=800, height=400):
    """
    Generate and display a word cloud from pandas Series of text
    """
    all_text = " ".join(text_series.tolist())
    wordcloud = WordCloud(width=width, height=height, background_color='white').generate(all_text)
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def extract_topics(text_series, n_topics=5, max_features=1000):
    """
    Perform LDA topic modeling and return top words per topic
    """
    from sklearn.decomposition import LatentDirichletAllocation
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    dt_matrix = vectorizer.fit_transform(text_series)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dt_matrix)
    
    topics = []
    for i, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
        topics.append(top_words)
    return topics
