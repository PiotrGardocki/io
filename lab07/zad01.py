import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text


with open('article.txt') as f:
    text = f.read()
    #text = pd.DataFrame({'reviewText': text})

#df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
#text = df['reviewText']
#df['reviewText'] = df['reviewText'].apply(preprocess_text)

for char in punctuation:
    text = text.replace(char, '')

tokens = word_tokenize(text.lower())
print(f'# of tokens: {len(tokens)}')
filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
print(f'# of non-stop tokens: {len(filtered_tokens)}')
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print(f'# of lemmatized tokens: {len(filtered_tokens)}')
print(lemmatized_tokens)

final_text = ' '.join(lemmatized_tokens)
wordcloud = WordCloud().generate(final_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

wordcloud = WordCloud(max_font_size=40).generate(final_text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

words = sorted(list(set(lemmatized_tokens)))
count = [(word, final_text.count(word)) for word in words]
count.sort(key=lambda t: t[1], reverse=True)
print(count)

df = pd.DataFrame({
    t[0]: t[1] for t in count
})

plt.figure(figsize=(15,10))
df.plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()
