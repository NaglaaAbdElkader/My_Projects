# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:34:48 2022

@author: Mohammad Joseph
"""

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples,stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist,classify,NaiveBayesClassifier

import re,string,random

def removie_noise(tweet_tokens,stop_words = ()):
    cleaned_tokens = []
    for token,tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","",token)
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token,pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens  

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token,True] for token in tweet_tokens)
        
if __name__ == "__main__":

    positive_tweeet = twitter_samples.strings('positive_tweets.json')
    negative_tweeet = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    
    stop_words = stopwords.words('english')
    
    positive_tweeet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweeet_tokens = twitter_samples.tokenized('negative_tweets.json')
    
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    
    for tokens in positive_tweeet_tokens:
      positive_cleaned_tokens_list.append(removie_noise(tokens,stop_words))
    for tokens in negative_tweeet_tokens:
      negative_cleaned_tokens_list.append(removie_noise(tokens,stop_words))
      
    all_positive_words = get_all_words(positive_cleaned_tokens_list)
    
    freq_dist_positive = FreqDist(all_positive_words)
    print(freq_dist_positive.most_common(10))
    
    model_positive_tokens = get_tweets_for_model(positive_cleaned_tokens_list)
    model_negative_tokens = get_tweets_for_model(negative_cleaned_tokens_list)
    
    positive_dataset = [(tweet_dict,"Positive") for tweet_dict in model_positive_tokens]
    negative_dataset = [(tweet_dict,"Negative") for tweet_dict in model_negative_tokens]
    dataset = positive_dataset + negative_dataset
    
    random.shuffle(dataset)
    
    train_data = dataset[:7000]
    test_data = dataset[7000:]
    
    classifier = NaiveBayesClassifier.train(train_data)
    
    print("Accuracy is:",classify.accuracy(classifier,test_data))
    
    print(classifier.show_most_informative_features(10))
    
    customer_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
    
    customer_tokens = removie_noise(word_tokenize(customer_tweet))
    
    print(classifier.classify(dict([token,True] for token in customer_tokens)))
