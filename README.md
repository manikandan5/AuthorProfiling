# AuthorProfiling

Authorship profiling is an important part of data
mining.There has been many attempts to learn about the author
of the text through subtle variations in the writing styles that
happens because of the difference in gender, age and social
groups.The information acquired in this way has a variety of
applications in the real world. This project involves identifying
the characteristics of the author of a particular tweet. The
demographics that we currently try to predict are age and sex.
We have used supervised machine learning approach to achieve
this goal. We extract the features from the training data and
then train the classifier on these features. The same process is
done for the test data and then the classifier classifies it into the
respective label. We ran the experiments against different feature
extractors and classifiers. The analysis on the results increased
our interest in the problem and we have tried linking the pattern
in each tweet and have made interesting observations connecting
the demographics and the tweet of the author.

# Project Structure
 1. getTweets.py - This file collects all the tweets from Twitter using tweepy-http://docs.tweepy.org/.
The file is also responsible to classify the label the tweets into their specific folders which is collectively
stored in the **processed-data folder**.
 
 2. insertIntoDb.py - This file is responsible for inserting the processed data into the database.
 We are using Mongo DB to store the data. The database is called **user-details** and the table has the following fields.
    > SNo - Status Number
    
    > StatusID - status
    
    > Status - tweet content
    
    > Age - Age range(18-24,25-34,35-49,50-64,65-xx)
    
    > Sex - Male/Female

 3. profiling.py - This file is responsible for forming the Training and Test data set, classification and training of the
 datasets and obtaining the results.
 
 # Commands
 To run the project -  python profiling.py 
 
 
