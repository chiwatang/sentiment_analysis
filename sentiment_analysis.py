# Capstone Project - Amazon Review Sentiment Analysis 
# Purpose is to utilise the power of NLP to sort Amazon Reviews into their respective sentiments

# First we import the modules we need, pandas to handle and manipulate our data
# Spacy to process and tokenize our reviews(sentences) 
# Into indivdiual tokens for our program to get an understanding of each word/token
# The spacytextblob libray will allow us to analyze the sentiment behind tokens 
# Collection of them in cognizant to tell us the sentiment

import spacy 
import pandas as pd
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob


# Select the language model (English) so sparse a sentence later
nlp = spacy.load("en_core_web_sm")

# Adds a new extension for we can work with doc 
# Tokens attributes in this case use it to evaluate the sentiment later on
nlp.add_pipe('spacytextblob')

# - DATA RETRIEVAL -
# Use pandas to create a dataframe from our amazon CVS file
amazon_reviews_df = pd.read_csv("Task_21/amazon_product_reviews.csv", sep = ",")


# - DATA CLEANSING / REPROCESSING -
# We have selected the most relevant columns as the other such 
# e.g time, reviewer name and product name id is not necessary
# Therefore we chose the ratings and recommended to check the sentiment of the reviews matches
relevant_columns = ["name","reviews.doRecommend","reviews.rating","reviews.text"]

# This is large dataset so limiting us to the first 100 helps 
# Reduce runtime but allows enough sample to check accuracy
relevant_reviews_df = amazon_reviews_df[relevant_columns].head(100)

# Remove any rows that are empty but our selected dataframe fortunately has no missing values
relevant_reviews_df.dropna(inplace = True)

# This is a function to remove stopwords that do not alter the meaning/sentiment of our reviews
# Decided against making reviews lowercase & removing punctuation as they could affect sentiment
def remove_stop_words(sentence): 

    ''' REmoves stop words from our sentence '''

    # Parse the sentence using spaCy 
    doc = nlp(sentence) 

    # Use a list comprehension to remove stop words 
    filtered_tokens = [token for token in doc if not token.is_stop] 

    # Join the filtered tokens back into a sentence 
    filtered_sentence = ' '.join([token.text for token in filtered_tokens])
  
    # Output our sentence 
    return filtered_sentence

# Now we use pandas to apply our function to our reviews columns in our dataframe
relevant_reviews_df.loc[:,"reviews.text"] = relevant_reviews_df["reviews.text"].apply(remove_stop_words)


# - DATA ANALYSIS -
# Here we have two functions the first to analyse the sentiment e.g postive, neutral or negative
def sentiment_analysis(sentence):

    ''' Analyses the sentiment of our sentence '''

    doc = nlp(sentence)

    # Analyze each word/token in our review and returns a float between [-1.0,1.0]
    sentiment = doc._.blob.polarity

    # Return our value
    return sentiment


# We create a list of our categorized sentiments 
sentiment_headers = ["Negative","Neutral","Positive"]

# Use pandas to apply our sentiment function to our reviews column and create a series
sentiment_scores = relevant_reviews_df["reviews.text"].apply(sentiment_analysis)

# Once we have a series we then insert it back into our df next to our review column
relevant_reviews_df.insert(4,"sentiment.scores",sentiment_scores,)

# This is the next step to determine whether the score are positive, neutral or negative
# This is a list to store our final sentiments
sentiment_list = []

# So for all our sentiments scores we just analyzed, we now categorized them
# Enumerate helps find the correct indexed score
for i,score in enumerate(sentiment_scores):

    # A score of 0 is perfectly neutral
    if score == 0:
        # We also add those sentiments to our list we just made
        sentiment_list.append(sentiment_headers[1])

    # A score of more than 0 is positive
    elif score > 0:
        sentiment_list.append(sentiment_headers[2])

    # A score of less than 0 is negative
    else:
        sentiment_list.append(sentiment_headers[0])

# Once we have a categorized all the sentiments with a complete list,
# We add the column to our df next to our scores
relevant_reviews_df["reviews_sentiment"] = sentiment_list

# To evaluate how accurate our analysis was, we can use the ratings provided
# We create a function to convert the a ratings into our previous categorized sentiments
def rating_sentiment(rating):

    ''' Converts rating into our categorized sentiment'''

    if rating < 3:
        altered_ratings = sentiment_headers[0]

    elif rating == 3:
        altered_ratings = sentiment_headers[1]
    
    elif rating > 3:
        altered_ratings = sentiment_headers[2]
    
    return altered_ratings

# Like before we add it as a column next to our other previous colums
relevant_reviews_df.loc[:,"rating_sentiment"] = relevant_reviews_df["reviews.rating"].apply(rating_sentiment)

# Now we can create a series to check whether our NLP sentiment was correct 
# By compareing to the ratings sentiment by the user
sentiment_comparison = np.where(relevant_reviews_df["reviews_sentiment"] == relevant_reviews_df["rating_sentiment"], True, False)

# We also add the series back into our DF to compare
relevant_reviews_df["sentiment_accurate"] = sentiment_comparison

# - EVALUATION / ANALYSIS OF RESULTS
# Gives us a break down of the sentiment results 
print(''' 

Breakdown of Our Sentiment Results
''')
print(relevant_reviews_df["reviews_sentiment"].value_counts())

print(''' 

                                    A Simple Table of Results With Our Results Columns 
''')
print(relevant_reviews_df)

print(''' 

Breakdown of Our How Accurate Our Results Were
''')

# Lets see a breakdown of our accuracy and how we did
print(relevant_reviews_df["sentiment_accurate"].value_counts())

# We can calcuate how accurate our sentiment analysis was by
# Diving True matching results to the total which was 100
results_accuracy = 100 *((relevant_reviews_df["sentiment_accurate"]== True).sum() / relevant_reviews_df["sentiment_accurate"].count())

# Prints to user our accuracy
print(''' 
Comparison of Data Rating Sentiment VS Our Predicted Review Sentiment
      ''')
print(f"We were {results_accuracy}% Accurate")