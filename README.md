# Text-Classfication-Neural-Network
Tensorflow trained neural network to determine whether movie reviews are positive or negative. 

### My personal review for a movie as a text file
![MovieReview](https://user-images.githubusercontent.com/106715980/178888250-c5028fde-603a-4589-846c-c9826f707fe7.png)
[MyReview.txt](https://github.com/Aiden-Robinson/Text-Classfication-Neural-Network/files/9108176/MyReview.txt)

### The output from running the model
![Just The Review](https://user-images.githubusercontent.com/106715980/178889667-b650683e-d186-4d8c-ab3c-9e514e6ce568.png)

## Training the model

## Preprocessing the Data
For both the data that the model was trained on and our individual review, there requires preprocessing on the .txt files inro a readable format for the model. When training the model, I set the vocabulary size to 10000 meaning it only recognizes the 10000 most common words that appear in the training data and any uncommon word that might only come up once or twice is marked as unknown so it doesn't throw the model off. The words and integers are placed in a dictionary called `word_index`, but I added 3 of my own keys first:
- `word_index["<PAD>"]= 0` is an integer to fill extra space in the review to meet a certain length requirement 
- `word_index["<START>"]=1` is an integer put at the start of every review to indicate that it is the start
- `word_index["<UNK>"]=2` is an integer used for unknown words (words not within the dictionary)
- `word_index["<UNUSED>"]` is used for unknown words


<img width= "500" height= "450" src= "https://user-images.githubusercontent.com/106715980/178895195-3f9081a1-9df7-4d1c-abc6-085c6fe58b00.png">
for example,this is how the review I wrote looks when it is encoded:

## Model architecture
