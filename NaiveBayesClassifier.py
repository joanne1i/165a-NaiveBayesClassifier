import numpy as np
import sys
import time
import resource


# returns file's documents
def getReviews(filename):
    reviews = []
    temp = []
    with open(filename, 'r') as fp:
        for line in fp:
            arr = line.split(" ,", 1)
            temp = [b for b in zip(arr[0].split(" ")[:-1], arr[0].split(" ")[1:])]
            # for b in zip(arr[0].split(" ")[:-1], arr[0].split(" ")[1:]):
            #     temp.append(b)
            reviews.append(temp)
            temp = []
    return reviews

def getSeparateReviews(filename):
    pos_count = 0
    neg_count = 0
    pos_reviews = []
    neg_reviews = []
    n_temp = []
    p_temp = []
    with open(filename, 'r') as fp:
        for line in fp:
            arr = line.split(" ,", 1)
            if('0' in arr[1]):
                neg_count += 1
                n_temp = [b for b in zip(arr[0].split(" ")[:-1], arr[0].split(" ")[1:])]
                neg_reviews.append(n_temp)
                n_temp = []
            else:
                pos_count += 1
                p_temp = [b for b in zip(arr[0].split(" ")[:-1], arr[0].split(" ")[1:])]
                pos_reviews.append(p_temp)            
                p_temp = [] 
    return neg_reviews, neg_count, pos_reviews, pos_count
    
# return each label's documents without stop words
# def getSeparateReviews(filename):
#     pos_count= 0
#     neg_count = 0
#     pos_reviews = []
#     neg_reviews = []
#     pos_reviews_t = []
#     neg_reviews_t = []
#     n_temp = []
#     p_temp = []
#     with open(filename, 'r') as fp:
#         for line in fp:
#             arr = line.split(" ,", 1)
#             if('0' in arr[1]):
#                 neg_reviews_t.append(arr[0])
#                 neg_count += 1
#             else:
#                 pos_reviews_t.append(arr[0])
#                 pos_count += 1
#     for l in neg_reviews_t:
#         for b in zip(l.split(" ")[:-1], l.split(" ")[1:]):
#             n_temp.append(b)
#         neg_reviews.append(n_temp)
#         n_temp = []
#     for l in pos_reviews_t:
#         for b in zip(l.split(" ")[:-1], l.split(" ")[1:]):
#             p_temp.append(b)
#         pos_reviews.append(p_temp)            
#         p_temp = []
#     return neg_reviews, neg_count, pos_reviews, pos_count


# returns all labels of file
def getLabels(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            if('0' in line):
                yield 0
            elif('1' in line):
                yield 1

# def getLabels(filename):
#     labels = []
#     with open(filename, 'r') as fp:
#         for line in fp:
#             if('0' in line):
#              labels.append(0)
#                 # yield 0
#             elif('1' in line):
#                 labels.append(1)
#                 # yield 1
#     return labels

# turns 2d list into 1d
def flatlist(list):
    newList = []
    # newList = [i for sublist in list for i in sublist]
    for sublist in list:
        for i in sublist:
            newList.append(i)
    return newList

# returns number of times each word shows up in file
def getDict(reviews): 
    dict = {}
    for word in reviews:
        dict[word] = dict.get(word, 0) + 1
    return dict

# vocabulary = total number of all unique words in file
def getVocabCount(filename):
    combined_vocab_count = 0
    combined_vocab = getReviews(filename)        
    combined_vocab_flat = flatlist(combined_vocab)        
    combined_dict = getDict(combined_vocab_flat)
    return len(combined_dict), combined_vocab

# returns training documents w/ no label
neg_reviews, neg_count, pos_reviews, pos_count = getSeparateReviews(sys.argv[1])
# neg_reviews, neg_count, pos_reviews, pos_count = getSeparateReviews("train.txt")

def prior_probability(neg_count, pos_count):
    # calculate prior probabilities for each label first
    total_docs = neg_count + pos_count
    neg_prior = np.abs(float(neg_count/total_docs))
    pos_prior = np.abs(float(pos_count/total_docs))
    return neg_prior, pos_prior

def getLabelDict(neg_reviews, pos_reviews):
    n = flatlist(neg_reviews)
    neg_dict = getDict(n)
    p = flatlist(pos_reviews)
    pos_dict = getDict(p)
    return neg_dict, pos_dict

neg_dict, pos_dict = getLabelDict(neg_reviews, pos_reviews)

# sum of total word counts in dict 
n = 0
pn = 0
for w, count in neg_dict.items():
    n += count
neg_n = n

for w, count in pos_dict.items():
    pn += count
pos_n = pn

def conditional_probability(label, word, vocab):
    alpha = 1
    # using training dictionary to find cond prob
    if(label == 0):
        dict = neg_dict
        n = neg_n
    else:
        dict = pos_dict
        n = pos_n
    # get count of word
    nk = dict.get(word)
    if(nk == None): nk = 0
    cond_prob = np.log(float(nk + alpha)/(n + alpha*np.abs(vocab)))
    return cond_prob

def classify(filename):
    C = [0, 1]
    vocab, documents = getVocabCount(filename)
    correct = 0
    incorrect = 0
    real_labels = getLabels(filename)
    neg_prior, pos_prior = prior_probability(neg_count, pos_count)
    # loops through all documents in file
    for index, document in enumerate(documents):
        # for each label (0 or 1)
        for cj in C:
            prob_sum = 0
            # for each word in document
            for word in document:
                # get the probability sum of the whole document
                # print(word)
                prob_sum += conditional_probability(cj, word, vocab)
            if(cj == 0):
                neg_total = np.log1p(neg_prior) + prob_sum
            else:
                pos_total = np.log1p(pos_prior) + prob_sum
        if(neg_total > pos_total):
            cnb = 0
        else:
            cnb = 1
        # if(cnb == real_labels[index]):
        if(cnb == next(real_labels)):
            correct += 1
        else:
            incorrect += 1
        # if(filename != sys.argv[1]):
            # print(cnb)
    return (correct/float(correct + incorrect))

# classify("train.txt")
start_time = time.time()         
train = classify(sys.argv[1])
train_t = int(time.time() - start_time)

start_time = time.time()         
test = classify(sys.argv[2])
test_t = int(time.time() - start_time)


print("{} seconds (training)".format(train_t))
print("{} seconds (labeling)".format(test_t))

print("{:.3f} (training)".format(train))
print("{:.3f} (testing)".format(test))

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)