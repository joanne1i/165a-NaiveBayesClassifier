import numpy as np
import sys

def getLabels(filename):
    labels = []
    count = 0
    with open(filename, 'r') as fp:
        for line in fp:
            if('0' in line):
                labels.append(0)
                count +=1
            elif('1' in line):
                labels.append(1)
                # count +=1
    print(labels)
    return labels
getLabels("train.txt")

def getReviews(filename):
    count = 0
    reviews = []
    with open(filename, 'r') as fp:
        for line in fp:
            arr = line.split(" ,", 1)
            arr = arr[0].split(" ")
            reviews.append(arr)
            count += 1
    # print(reviews, count)
    return reviews

def getPositiveClass(filename):
    count = 0
    reviews = []
    with open(filename, 'r') as fp:
        for line in fp:
            if('1' in line):
                arr = line.split(" ,", 1)
                arr = arr[0].split(" ")
                reviews.append(arr)
                count += 1
    # returns a tuple, have to access through result[0] 
    return reviews, count  

# separate label from reviews and split sentences into words
def getNegativeClass(filename):
    count = 0
    reviews = []
    with open(filename, 'r') as fp:
        for line in fp:
            if('0' in line):
                arr = line.split(" ,", 1)
                arr = arr[0].split(" ")
                reviews.append(arr)
                count += 1
    # returns a tuple, have to access through result[0] 
    return reviews, count  

def flatlist(list):
    newList = []
    for sublist in list:
        for i in sublist:
            newList.append(i)
    return newList

def getDict(reviews): 
    dict = {}
    for word in reviews:
        dict[word] = dict.get(word, 0) + 1
    return dict

# extracted training data
training_neg = getNegativeClass("training.txt")
updated_rev = flatlist(training_neg[0])
neg_dict = getDict(updated_rev)

training_pos = getPositiveClass("training.txt")
updated_rev = flatlist(training_pos[0])
pos_dict = getDict(updated_rev)



# calculating prior probability P(cj)
C = ['0', '1']
total_docs = []
prior = []

for c_j in C:
    if(c_j == '0'):
        docs = training_neg[1]
    elif(c_j == '1'):
        docs = training_pos[1]
    total_docs.append(docs)
for doc_j in total_docs:
    prior.append(float(doc_j)/sum(total_docs))

def print_confusion_matrix(matrix, class_labels):
    lines = ["" for i in range(len(class_labels)+1)]
    for index, c in enumerate(class_labels):
        lines[0] += "\t"
        lines[0] += c
        lines[index+1] += c
    for index, result in enumerate(matrix):
        for amount in result:
            lines[index+1] += "\t"
            lines[index+1] += str(amount)
    for line in lines:
        print(line)

def initialize_conversion_matrix(num_labels):
    return [[0 for i in range(num_labels)] for y in range(num_labels)]

confusion_matrix = initialize_conversion_matrix(2)
combined_vocab = getReviews("training.txt")
combined_vocab = flatlist(combined_vocab)
combined_dict = getDict(combined_vocab)
print(combined_dict)
combined_vocab_count = len(combined_dict)


correct = 0
incorrect = 0
for c_j in C: # 0 or 1
    # extract testing data
    if(c_j == '0'): 
        testing_neg = getNegativeClass("testing.txt")
        updated_rev = flatlist(testing_neg[0])
        test_neg_dict = getDict(updated_rev)
        all_words = len(test_neg_dict)
        dict = test_neg_dict
        score = np.log(prior[0])
        i = 0
    else:
        testing_pos = getPositiveClass("testing.txt")
        updated_rev = flatlist(testing_pos[0])
        test_pos_dict = getDict(updated_rev)
        all_words = len(test_pos_dict)
        dict = test_pos_dict
        score = np.log(prior[0])
        i = 1
    scores = []
    # calculating conditional probability
    for index in range(0,2):
        score = np.log1p(prior[index])
        for word,count in dict.items():
            # conditional probability = number of times a word appears in all of the
            # documents in that class DIVIDED by the total number of words seen in 
            # that class     
            conditional_prob = float(count+1)/(all_words + combined_vocab_count)
            score += np.log(conditional_prob)
        scores.append(score)
    max_index, max_value = max(enumerate(scores), key=lambda p: p[1])
    confusion_matrix[i][max_index] += 1

    if i == max_index:
        correct += 1
    else:
        incorrect += 1
print(correct / float(correct + incorrect))
print(confusion_matrix)



