# Indy-Archieve-Document-Classification
![](Images/document-classifier-photo.png)

# Problem Statement
Document classification is a problem faced by many organizations globally. As documents increase in volume, their capacity to be stored multiplies and manual document annotation has become expensive, unscalable, and impractical. To rectify this problem, data scientists now employ many modes of algorithms with the goal of classifying these documents automatically.

Indianapolis is also not immune to this problem, and thus this project began in collaboration with The Polis Center of Indianapolis in order to solve this very problem: “How can we classify large numbers of text articles from the Encyclopedia of Indianapolis based on subject?”. 

The Encyclopedia of Indianapolis contains thousands of entries ranging from the subjects of bibliographic timepieces of influential individuals, to descriptive geography, and centuries of recorded events - all of which are currently unlabeled.

# Goal of this project
The goal of this project is to automate the document classification process by building a predictive model to classify them into categories of geography, time, person, and organization.

This categorization will facilitate archival and search of articles for years to come and get rid of the mundane tasks of reading and annotating articles manually.

# Why Deep Learning?
With the increase in corpus size, automation in classification is much needed. This is reason why deep learning algorithms out-perform other techniques if the data size is large.

Deep Learning typically requires a large quantity of training data to ensure that the network, which may very well have tens of millions of parameters and does not overfit the training data.

In this project, I used a deep neural network which employs a Bidirectional Long-Short-Term Memory network (Bi-LSTM) to classify Encyclopedia of Indianapolis articles.

# Data Source
Data for this project was generously provided by The Polis Center of Indianapolis. It consisted of roughly 1,600 plain-text articles in JSON format. A data frame was first made available through API and transformed in Python. It was then converted into a CSV format for further processing in Pandas.

The dataset contains the following features:
Body: The body of the document which contained the document’s content.
Category: The category to which the document belongs.

Label categories were classified as one of the following: “time”, “person”, “organization”, and “geography”,
Where “time” consisted of articles primarily describing years, dates, or eras. 
“Person” consisted of articles describing specific or small groups of individuals. 
“Organization” consisted of articles describing businesses, non-profits, not-for-profit, clubs, etc.
“Geography” consisted or articles describing developments, cities, towns, and land.

# Data Pre-Processing
1.	Dropping the duplicate rows:
For safety, any duplicate entries or missing values from the dataset were removed. 

2.	Removing the special characters from articles:
Punctuation was also removed using regular expression in Python, and type-case was lowered.

3.	Removed “run-on” words from articles: 
Upon inspecting the data, it became apparent that one additional step was necessary, which included filtering words longer than 15 characters. This was necessary due to articles which contained accidental “run-on” words, such as in the following example:
wasanextracurricularorganizationthatprepared 

One solution to this problem which should be considered is an algorithm which divides these words based on plausible spacing positions, however the implementation of this method falls outside of the scope of our research and can be taken a future to-do.
For this project, these words were removed from the dataset all-together. 

# Exploratory Data Analysis
1.	Distribution of different categories in the dataset.
The distribution of categories in the dataset was not uniform, which resulted in a larger number of articles about people and organizations, rather than time or geography.
![](Images/Distribution%20of%20classes.png)

2.	Finding the number of words in each document.
Articles ranged from a few sentences to several paragraphs, with a mean
length of 380 words. Percent of articles greater than 380 words.
![](Images/Average%20Number%20of%20words.png)

# Load and Prepare data for Modelling:





