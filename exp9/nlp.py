import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# reading and wragling data
df_avatar = pd.read_csv('avatar_data.csv', engine='python')
df_avatar_lines = df_avatar.groupby('book').count()
df_avatar_lines = df_avatar_lines.sort_values(by=['book_chapt'], ascending=False)[:10]
top_book_names = df_avatar_lines.index.values
# filtering out non-top characters
df_book_sentiment = df_avatar[df_avatar['book'].isin(top_book_names)]
df_book_sentiment = df_book_sentiment[['book', 'book_chapt']]
# calculating sentiment score
sid = SentimentIntensityAnalyzer()
df_book_sentiment.reset_index(inplace=True, drop=True)
df_book_sentiment[['neg', 'neu', 'pos', 'compound']] = df_character_sentiment['book_chapt'].apply(sid.polarity_scores).apply(pd.Series)
df_book_sentiment
