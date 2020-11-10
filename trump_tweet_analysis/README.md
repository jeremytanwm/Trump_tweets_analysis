# Capstone Project: Predicting the impact of president Trump’s tweets on US indices
#### Jeremy Tan Wei Ming DSI-17 Singapore
#### 15 October 2020

## Problem Statement

### Context
Stock market is always unpredictable. Banks and traders are always looking out to improve their trading strategies. Any information or indicators that able to give an edge or advantage is always sought after.
The President of United States, Donald Trump currently has 87.2 millions followers on Twitter under the handle of @realDonaldTrump. Lucky for traders, the President is highly active on his twitter account and isn't shy about tweeting his views basically on anything. By understanding the impact(positive/negative) of President Trump’s tweets has in the Stock market, this will improve one’s investing/trading strategies.

I will analyse President Trump's tweets beginning from his presidency in 2017. Using natural language processing(NLP) tools such as topic modelling LDA to identify tweets that are trade/market related and each tweet is then assigned a sentiment score which will be used in the modelling. I will build a classification model that will look into the text features of President Trump tweets and how it can be classified to have positive or negative impact on the market. This model can be use as an indicator for buy signals, we want to want have high accuracy on both true positive and negative therefore our success will be measured by the accuracy score.


### Goal of project
- Predict the stock market movement by analysing President Trump tweets
- Take advantage of this predictions to create a profitable trading strategy


### Types of model that will be developed
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Xgboost Classifier
- SVC

### Success Evaluation
- For prediction: Model that has the highest accuracy score


## Executive Summary

The general workflow of this project follows:

1. Data Collection and Preprocessing: 

President Trump tweets were collected from http://www.trumptwitterarchive.com/archive. A total of 23357 tweets from the day he became President, 21 Jan 2017 till 8 Oct 2020. This dataset consist of content of the tweet, date and time, favourite counts, retweet counts, source of tweet and lastly the unique id of each tweet. Tweets were cleaned by removing emojis, url and any characters that ain't normal english words/ letters. We want to look at tweets that were made by Trump hence we drop all the retweets. Also any tweets that contain less than 3 words were consider as noise were dropped as well. 

Next we extract the S&P 500 index data from yahoo finance which we will be using this as the benchmark of the US stock market. The US stock market open from Monday to Friday, 930am to 430pm. In order to classify if tweets have impact on the stock market, a label will be assigned to each tweet on how the stock market perform (percentage changes) on the next trading day (t+1). 1 will be positive and 0 for negative changes in the closing price. Example if President Trump tweeted on 2017-01-20 (friday), we will look at the percentage change of prices on 2017-01-23 (monday) and assign the label accordingly. 1 if the changes is positive else 0. 

Next, tweets were tokenize, words were made lowercase and punctuation are removed. Through the exploratory data analysis we were able to identify meaningless high frequency words. Such words are similar as stop words which occur in abundance, hence providing little to no unique information that can be used for classification. In order to have a better classification accuracy i will add those to the list of stop words and remove from the data. Words were then lemmatized to reduce each words to it's based form and is ready for topic modelling and sentiment analysis. 
f

2. Topic Modelling LDA

After preprocessing, the words are passed into topic model. Topic modeling is a type of statistical modeling for discovering the abstract “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

After many iterations and subjective judgement about the topics, it brings me to the final 6 topics. I used pyLDAvis help me interpret the topics in the topic model that has been fit to a corpus of text data. Topic 1 was capture words relevant to trade and economy. The model assigned a probability distribution of topics for a given tweet and since we are only focusing on Trade/economy, other topics were dropped. 


3. Sentiment analysis with VADER

Sentiment Analysis, is a sub-fielsd of Natural Language Processing (NLP) that tries to identify and extract opinions within a given text. The aim of sentiment analysis is to gauge the attitude, sentiments, evaluations, attitudes and emotions of a speaker/writer based on the computational treatment of subjectivity in a text. I will be using VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze Presdient Trump tweets. Each tweets will returns a positive, neutral, negative, and compound sentiment score. To identify tweets that have the greatest ability to move the market in the short-term, tweets with substantial positive (+0.7) or negative sentiment (-0.7) were isolated. Tweets that does not meet the cutoff were dropped.


4. Modelling

After comparing the 5 models, MultinomialNB has the best overall score. All models seems to be slightly overfitted. The test score for all 5 models hover around 0.8 and MNB has the lowest false positives which is important when we deploy this model for trading. Trading on false positives might result financial loss. Hence we rather missed good trade than to trade on false signals. 

After Gridsearch for the best parameters, MultinomalNB have an accuracy of 0.85 on test set with a ROC score of 0.87. 

|                    | log_reg | nb     | rf    | xgb    | svc    |
|--------------------|---------|--------|-------|--------|--------|
|01 Train score      |0.9830   |0.9490  |0.9908 |0.9765  |1.0000  |
|02 Test score       |0.8281   |0.8490  |0.8281 |0.8490  |0.8542  |
|03 Score diff.      |0.1549   |0.1     |0.1627 |0.1275  |0.1458  |
|04 Precision        |0.8516   |0.8742  |0.8114 |0.8343  |0.8519  |
|05 Specificity      |0.5400   |0.6200  |0.3400 |0.4400  |0.5200  |
|06 Sensitivity      |0.9296   |0.9296  |1.0000 |0.9930  |0.9718  |
|07 F1 Score         |0.8889   |0.9010  |0.8959 |0.9068  |0.9079  |
|08 True Negatives   |27       |31      |17     |22      |26      |
|09 False Positives  |23       |19      |33     |28      |24      |
|10 False Negatives  |10       |10      |0      |1       |4       |
|11 True Positives   |132      |132     |142    |141     |138     |
|12 Train ROC Score  |0.9982   |0.9844  |1.0000 |0.9993  |1.0000  |
|13 Test ROC Score   |0.9190   |0.8732  |0.9117 |0.9030  |0.9465  |


5. Backtesting Trading Strategy (Recommendation)

Based on the predictions, i have created a simple trading strategy to backtest on the S&P500. We will be focusing only on buy signals for this strategy. 1 indicating buy signal and 0 to do nothing. The strategy is to buy at tomorrow's open and sell at tomorrow’s close if the tweet is predicted 1. 

The trading strategy based on our prediction model returned a 72% vs a buy and hold strategy of 52% over the same period of time. We also observed that the returns of the strategy was behind until around March 2020 when there's a heavy selloff in the market. Even thou returns was lower but this strategy greatly reduce the volatility in the portfolio. Having a buy and hold strategy means that you expose yourself to the 100% volatility in the market. You can clearly see that during selloff in March 2020, not only our strategy was not impacted by the  selloff but on the other hand we made around 20%.

However we also need to be aware that there's a possibility that we are just lucky. Majority of the buy signals were generated in market uptrends, in a trending bull market it will generate more buy signals and President Trump tweets might just coincide with other market mover such as non farm payroll events, FED economic data release, or other factors. For this project we will state that his tweets does have a positive relationship with market returns. 


## Conculsion

In conclusion, through topic modelling with LDA and VADER sentiment analysis we were able to filter out the noise of President Trump tweets and isolate his sentiment towards trade and economy related issues. Using a MultinomialNB model to classify President Trump tweets that have positive impact on the market and use those predictions over a backtest was profitable and outperform the buy and hold strategy.










