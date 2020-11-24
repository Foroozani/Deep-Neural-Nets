# Emojify

To use word vector representations of sequence models to build an Emojifier

Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!" the emojifier can automatically turn this into "Congratulations on the promotion! +1 Lets get coffee and talk. coffee Love you! heart"

![](https://github.com/Foroozani/Neural-Nets-Deep-Learning/blob/main/sequence_models/images/data_set.png)



You will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (baseball). In many emoji interfaces, you need to remember that heart is the "heart" symbol rather than the "love" symbol. But using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular emoji, your algorithm will be able to generalize and associate words in the test set to the same emoji even if those words don't even appear in the training set. This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set.

In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings, then build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM.


Baseline model: Emojifier-V1 1.1 - Dataset EMOJISET Let's start by building a simple baseline classifier. You have a tiny dataset (X, Y) where:

*X* contains 127 sentences (strings) *Y* contains a integer label between 0 and 4 corresponding to an emoji for each sentence (refer image)

**Expected Output:**

Train set accuracy 97.7 Test set accuracy 85.7 Random guessing would have had 20% accuracy given that there are 5 classes. This is pretty good performance after training on only 127 examples.

In the training set, the algorithm saw the sentence "I love you" with the label heart. You can check however that the word "adore" does not appear in the training set. Nonetheless, lets see what happens if you write "I adore you."


If you have an NLP task where the training set is small, using word embeddings can help your algorithm significantly. Word embeddings allow your model to work on words in the test set that may not even have appeared in your training set. Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details: To use mini-batches, the sequences need to be padded so that all the examples in a mini-batch have the same length. An Embedding() layer can be initialized with pretrained values. These values can be either fixed or trained further on your dataset. If however your labeled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings. LSTM() has a flag called return_sequences to decide if you would like to return every hidden states or only the last one. You can use Dropout() right after LSTM() to regularize your network. 



**Refrence**

 Andrew NG class on Sequence Models: https://www.coursera.org/learn/nlp-sequence-models
