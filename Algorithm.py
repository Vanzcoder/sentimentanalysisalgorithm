
import numpy as np
from sklearn.utils import shuffle

fp = open("AppleReviews.csv","r")

temp_data =[]
data = []
labels= []

for line in fp:
    review = fp.readline()
    review = review.split(',')
    temp_data.append(review)

shuffle_data = shuffle(temp_data, random_state= 42)

for line in shuffle_data: 
    labels.append(line[0])
    data.append(line[1].split())

total = len(data)
train = int(total*.7)
train_data = data[:train]
train_labels = labels[:train]
test_data = data[train:]
test_labels = labels[train:]
print(len(test_data))


def clean_data(train_data, test_data):
    for line in train_data:
        for i in range(len(line)):
            line[i] = line[i].strip("#")
            line[i] = line[i].strip("!")
            line[i] = line[i].strip("@")
            line[i] = line[i].strip("?")
            line[i] = line[i].lower()
    for line in test_data:
        for i in range(len(line)):
            line[i] = line[i].strip("#")
            line[i] = line[i].strip("!")
            line[i] = line[i].strip("@")
            line[i] = line[i].strip("?")
            line[i] = line[i].lower()
              
    return train_data, test_data

#total word count in positive and negative tweets
def word_count(train_data):
    count_pos = 0
    count_neg = 0
    for i in range(len(train_labels)):
        if train_labels[i] == "pos" or train_labels[i] == "Pos":
            count_pos+=len(train_data[i])
        else:
            count_neg+=len(train_data[i])
    return count_pos, count_neg
    
def probs(train_data, train_labels, count, total_count, word, sentiment):  
    counter=0
    for i in range(len(train_data)):
        #only counts the word if the sentiment (in the tweet) matches the sentiment (in the function like "pos" or "neg")  
        if train_labels[i] == sentiment:
            for w in train_data[i]:
                #how much that word occurs
                if w == word:
                    counter+=1
    #conditional_prob is the probabily given the sentiment
    conditional_prob = (counter+1)/(count+1)
   
    #prior prob is the probability in general
    counter2 = 0
    for i in range(len(train_data)):
        for w in train_data[i]:
            #how much that word occurs
            if w == word:
                counter2+=1
    
    #the +1 is to prevent a zero from messing up all of the probabilities
    prior_prob = (counter2+1)/(total_count +1)
            
    #probability = conditional_prob * pior_prob
    prob = conditional_prob*prior_prob
    return prob
        
        
        
def naivebayes_classifier(test_data, train_data):
    count_pos, count_neg = word_count(train_data)
    total_count = count_pos + count_neg
    for line in test_data:
        pos_prob = 1
        neg_prob = 1
        for word in line:
            neg_prob *= probs(train_data, train_labels, count_neg, total_count, word, "neg")
            pos_prob *= probs(train_data, train_labels, count_pos, total_count, word, "pos")
        if pos_prob>neg_prob:
            print("positive: ", pos_prob, neg_prob)
        else:
            print("negative: ", neg_prob, pos_prob)
       
#calling the functions
clean_train, clean_test = clean_data(train_data, test_data)
naivebayes_classifier(clean_test, clean_train)

#the probability for each word is small because each is very unique
