# FINE-TUNE-BERT((Bidirectional Encoder Representations from Transformers)-FOR-TEXT CLASSIFICATION-WITH-TENSERFLOW
//Why we use BERT 
---It's is a pre-trained modle availablr in TRANSFORMARE Library of rensoflow that help categories like positive/bnegetive ect.
//NLP stands for Natural Language Processing. It gives the ability to machine understand and process human languages. Human languages can be in the form of text or audio format.
----advantage - NLP helps us to analyse data from both structured and unstructured sources.
NLP is very fast and time efficient.
----disadvantage - NLP  require huge data 
-----APPLICATION - text classification , audio processing 

//Tensorflow - it is an ipen source libery for ml and deep ai,that provide the extensive tools for NLP.
---use for tenssorflow for NLP 
1.Tokenization processing 
2.sentimaent analiysing 
3.Text generations 

//WE are using Tensor flow 2 - 
1.Enabled by default (dynamic, imperative execution).
2.Simplified APIs and more Pythonic code.
3.Keras is the central high-level API (tf.keras).
4.Easier debugging with eager exe

CREATE A DATASET FOR TARNNIGN AND EVALUATION -
step 1 - train_test_split -A function from the sklearn.model_selection module use for spliting the dataset in test , traning and validation set of size .0075 for train set and 0.00075 for validation set .
step 2 - Conversion to TensorFlow Datasets:
Pandas DataFrame values (question_text and target) are converted into TensorFlow's
step 3 - tf.device('/cpu:0'):Ensures the preprocessing is done on the CPU,
step 4 - take Method:Fetches a limited number of items from the dataset for quick inspection.


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

tokenisation - it's  be hard to understand the sentiment through letter tokenisation (listen ans silent ) same letters 
so go for word tokenisation 
i love my dog (001 , 002 , 003 , 004) i love my cat (001 , 002 , 003 , 005)
and for encoding these sentence maitain the length by masking (adding 0)

and then use sigmoid fynction to get the value b/w 0-1 and set the treshold value then classify this ,
 traning and testing

 //is neural network creat a words (for generating text we don.t need validation data prediction)
 recurrent neural networking - recomending best data (word) that fit the missing word of the sentence 
          it is like febunaci series

recurremnt neural network -
