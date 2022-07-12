import tensorflow as tf #importing all the libraries we need
from tensorflow import keras
import numpy as np

data= keras.datasets.imdb #importing our dataset

(train_data, train_labels), (test_data, test_labels)= data.load_data(num_words= 10000)# separate data into test and train

print(train_data[0]) #print out an example review

word_index= data.get_word_index() #gives us tuples that have the integer and corresponding word with them

word_index= {k:(v+3) for k,v in word_index.items()} #breaks up the tuple into keys and values for a dictionary, the +3 is to add some of our own keys
word_index["<PAD>"]= 0 #padding is used to fill extra spaces in a review to meet a certain length requirment
word_index["<START>"]=1 #this is for the start of every review so we know its the start
word_index["<UNK>"]=2 #for unknown words in the review, or words less common than the top 10000
word_index["<UNUSED>"]=3 #for unused words

reverse_word_index= dict([(value, key) for (key, value) in word_index.items()]) #reversing the order of keys and valyes


model= keras.models.load_model("model.h5")# loading in the model

def review_encode(string):
    encoded= [1]

    for word in string: #for each word
        if word.lower() in word_index: #if the word is in our dictionary, (passes everything to lower case though)
            encoded.append(word_index[word.lower()]) #add the numerical index of the word to our list
        else:
            encoded.append(2) #if it doesnt recognize the word, append the tag for unkown
    return encoded

with open("MyReview.txt", encoding= "utf-8") as f: #so we dont have to close the text, the utf-8 is just a constant
    for line in f.readlines(): #reading line by linein our review
        nline= line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ") #trimming unnecessary charaters so we have only words
        encode= review_encode(nline) #encoding the current line
        encode= keras.preprocessing.sequence.pad_sequences([encode], value= word_index["<PAD>"], padding= "post", maxlen= 250) #preprocessing the current line
        predict= model.predict(encode) #predicting the current line
        print(line) #printing the line in text
        print(encode) #printing the line in numerical indices
        #print(predict[0]) #printing the preiction value
        if predict[0][0]>0.5: #printing the accuracy
            print("The review was postive with",100*predict[0][0],"percent accuracy")
        elif predict[0][0]<0.5:
            print("The review was negative with",100-(100*predict[0][0]),"percent accuracy")
        else:
            print("The movie was neutral")


