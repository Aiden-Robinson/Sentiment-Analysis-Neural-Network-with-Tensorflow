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

train_data= keras.preprocessing.sequence.pad_sequences(train_data, value= word_index["<PAD>"], padding= "post", maxlen= 250) #padding the reviews, padding takes place after and max words is 250
test_data= keras.preprocessing.sequence.pad_sequences(test_data, value= word_index["<PAD>"], padding= "post", maxlen= 250)

print(len(train_data[4]), len(test_data[6])) #sample printing the length of 2 reviews

def decode_review(text): #function that decodes the text from integers to strings, question mark is for unkown words
    return " ".join([reverse_word_index.get(i, "?") for i in text])


model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16)) #creating 10000 word vectors with 16 dimensions
model.add(keras.layers.GlobalAveragePooling1D()) #flattening our 16 dimensions into a lower dimensions
model.add(keras.layers.Dense(16, activation="relu")) #activation function for our dense layers
model.add(keras.layers.Dense(1, activation="sigmoid")) #function that will squish our value between 0 and 1. we want this to determine wh sether positive or negative review

model.compile(optimizer= "adam", loss= "binary_crossentropy",metrics= ["accuracy"])#choosing binary_crossentropy because we have 2 options for our output neuron (0 or 1)

x_val= train_data[:10000] #splitting up into validation data and training data
x_train= train_data[10000:] #validation data checks how well our model is performing on new data

y_val= train_labels[:10000]
y_train= train_labels[10000:]

fitModel= model.fit(x_train, y_train, epochs= 40, batch_size=512, validation_data= (x_val, y_val), verbose=1) #batch size menas reading in 512 movie reviers at once, run this 40 times

results= model.evaluate(test_data, test_labels)


model.save("model.h5")

