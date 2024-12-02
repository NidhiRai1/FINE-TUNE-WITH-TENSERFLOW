# FINE-TUNE-BERT((Bidirectional Encoder Representations from Transformers)-FOR-TEXT CLASSIFICATION-WITH-TENSERFLOW
//Why we use BERT 
---It's is a pre-trained modle availablr in TRANSFORMARE Library of tensoflow that help categories like positive/bnegetive ect.
//NLP stands for Natural Language Processing. It gives the ability to machine understand and process human languages. Human languages can be in the form of text or audio format.
----advantage - NLP helps us to analyse data from both structured and unstructured sources.
NLP is very fast and time efficient.
----disadvantage - NLP  require huge data 
-----APPLICATION - text classification , audio processing 

//Tensorflow - it is an open source libery for ml and deep ai,that provide the extensive tools for NLP.
---use for tenssorflow for NLP 
1.Tokenization processing 
2.sentimaent analiysing 
3.Text generations 

//When to Use:
Use TensorFlow if:

You want to build and train custom models from scratch.
Your use case requires a highly customized solution.
Use TensorFlow-Hub if:

You want to leverage pre-trained models for quick implementation.
You're working on transfer learning tasks and need to save time and resources.

IMPORT THE QUORA INSTATNACE DATASET 
Why Would You Use Eager Mode in BERT Sentiment Analysis?
Debugging: You might enable eager mode while building or fine-tuning a BERT model to inspect tokenized inputs, embeddings, or outputs dynamically.
Development: Helps during prototyping or experimentation stages when you're testing preprocessing steps or tweaking hyperparameters.

CREATE A DATASET FOR TARNNIGN AND EVALUATION -
step 1 - train_test_split -A function from the sklearn.model_selection module use for spliting the dataset in test , traning and validation set of size .0075 for train set and 0.00075 for validation set .
step 2 - Conversion to TensorFlow Datasets:
Pandas DataFrame values (question_text and target) are converted into TensorFlow's
step 3 - tf.device('/cpu:0'):Ensures the preprocessing is done on the CPU,
step 4 - take Method:Fetches a limited number of items from the dataset for quick inspection.

DOWNLOAD A PRE TRAINNED BERT MODEL FROM TENSORFLOW HUB
Data preprocessing consists of transforming text to BERT input features:
input_word_ids - A token id corresponde to a patcular token
input_mask - to indicatr which is a token and which is a pedding
segment_ids - understand sentence A and B in a pair of sentance like if else etc.
--verval file mapping between words and their unique ids 
--lowercasing help the model to treat likeHI is same as hi 

 Tokenizing a Sentence - "hi, how are you doing?", as ['hi', '##,', 'how', 'are', 'you', 'doing', '##?']

 Converting Tokens to IDs - [7632, 29623, 2129, 2024, 2017, 2725, 29632]

 TOKENIZE A PRE-PROCESS TEXT FOR BERT

What tf.py_function Does
tf.py_function bridges the gap between graph mode and eager mode:

It allows TensorFlow to execute Python code (to_feature) within a computational graph.
Converts TensorFlow tensors to regular Python tensors (with .numpy() access) for the wrapped function, then maps the results back into TensorFlow tensors.

Sentiment Analysis for tensorflow -
pyhton code import and  dataset sample 
step 1 - Text Tokenization - covert sentence into numberic sequence
       "i like eating food" into [1,2,3,4]
step 2 - padding - ensuring each sequence of same length 
         [1,2,3] into [1,2,3,4,0,0,0]
step 3 - embedding - it convert the o/p got after paddign into a dense vector that allow
         the modle to understanding the relationship like king is closs to queen not to hair 
         [[0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9],
          [1.0, 1.1, 1.2]]
step 4 - LSMT (long short team memory) it do the word processing and give the result in sequensial order 
         Text has sequential patterns (e.g., "not good" means negative, even though "good" is positive in isolation).
step 5 - Denselayer - it mark the LSTm o/p in a single one by the help of sigmoid activation function 
         it will help to get the o/p in between 0 and 1.
         Eg. Probability = sigmoid(w1*0.3 + w2*0.4 + w3*0.5 + bias)
step 6 - Tranning - This will help to train the weight sof embbeded , LSTm and dense layer to reduce the losses .
         eg. "I love this product" → Positive (1)
 

//TOKENIZATION - splliting 
-----sentence tokenizer
from nltk.tokenize import sent_tokenize
text = "Hello everyone. Welcome to GeeksforGeeks. You are studying NLP article"
sent_tokenize(text) 
-----word tokenizer 
word_tokenize
-----white spcae tokenizar
from nltk.tokenize import WhitespaceTokenizer
tk = WhitespaceTokenizer()
gfg = "GeeksforGeeks \nis\t for geeks"
geek = tk.tokenize(gfg)
print(geek)

-----regular expression tokenize
Syntax : tokenize.RegexpTokenizer()
Return : Return array of tokens using regular expression


tokenisation - The BERT tokenizer is a crucial component of the BERT (Bidirectional Encoder Representations from Transformers) model. 
It converts input text into a numerical format suitable for the model. The tokenizer breaks down the text into smaller units called tokens and maps them to corresponding IDs in the model's vocabulary. 
i love my dog (001 , 002 , 003 , 004) i love my cat (001 , 002 , 003 , 005)
and for encoding these sentence maitain the length by masking (adding 0)

INPUT MASK - Input masking in BERT plays a crucial role in ensuring that the model processes only meaningful input tokens and ignores padding tokens.

and then use sigmoid fynction to get the value b/w 0-1 and set the treshold value then classify this ,
 traning and testing
