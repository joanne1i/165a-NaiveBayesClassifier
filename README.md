# Naive Bayes Classifier 
## Architecture
I wrote different functions to make applying the Naive Bayes’ algorithm a lot easier. The functions include separating documents from labels, 
separating positive reviews and negative reviews, setting up the dictionary count, counting the total number of unique words, etc. 
I ended up storing the reviews, labels and dictionaries in lists. I also wrote out the prior probability and conditional probability functions 
before classifying. The prior probability is calculated for each label (0 and 1). For each class, it takes the number of documents and divides it 
by the sum of the total documents. I followed the equation for conditional probability from the slides. The classify function takes each document 
within the file, takes each word within the document and calculates the probability sum of each word. When it finishes reading the document, it adds 
the total probability sum to the prior probability to get either neg_total or pos_total. It classifies the document as either 1 or 0 depending on which 
total is larger. It does this process until the whole file is read.

## Preprocessing
For preprocessing, I decided to use the bag-of-words model. Since the punctuation was already removed, it made it easier to implement this. 
The bag-of-words model is used in methods of document classification where the occurrence of each word is used as a feature for training a classifier. Each dictionary 
for each label ended up in the format of: {‘the’: 134, ‘bad’: 60 ... }. In my submission, I did not implement other transformations or additional features.

## Model Building
I ended up using the learning algorithm provided in the slides for the Naive Bayes’ implementation. I calculated the likelihood of each feature by finding the conditional probability and using the dictionary.
The conditional probability is calculated following the equation: (n<sub>k</sub> + a)/(n + α)*|Vocab|, where n<sub>k</sub> is equal to the
count of each word within the dictionary, α is used for smoothing, n is total count of words in the
dictionary, and Vocab is total number of unique words in the file. I implemented smoothing to account for the words that show up in the file, 
but do not exist in the dictionary. I ended up setting alpha as 1 because I saw that sklearn uses alpha = 1 as their default. I changed it during 
testing to see what would happen, but I did not notice much of an improvement. Multiplying lots of probabilities, which are between 0 and 1 by definition, 
can result in floating-point underflow. To combat it, I used the log version of the formula, which gave me better results.
