# Text-Classfication-Neural-Network
Tensorflow trained neural network to determine whether movie reviews are positive or negative. Followed the instructions on Tensorflows website to complete the project. 

### My personal review for a movie as a text file
![MovieReview](https://user-images.githubusercontent.com/106715980/178888250-c5028fde-603a-4589-846c-c9826f707fe7.png)

### The output from running the model
![Just The Review](https://user-images.githubusercontent.com/106715980/178889667-b650683e-d186-4d8c-ab3c-9e514e6ce568.png)

## Training the model

## Preprocessing the Data
For both the data that the model was trained on and our individual review, there requires preprocessing on the .txt files inro a readable format for the model. When training the model, I set the vocabulary size to 10000 meaning it only recognizes the 10000 most common words that appear in the training data and any uncommon word that might only come up once or twice is marked as unknown so it doesn't throw the model off.  
