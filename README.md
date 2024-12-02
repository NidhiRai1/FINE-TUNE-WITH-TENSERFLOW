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

TOKENIZE A PRE-PROCESS TEXT FOR BERT--------
We need to transform raw data (text and label) into a format that BERT understands, which consists of:

input_ids: Token IDs generated from the text.
input_mask: Specifies which parts of the input are real data and which are padding.
segment_ids: Distinguishes sentence pairs (if present) in the input.
label_id: Numeric representation of the label.

What tf.py_function Does
tf.py_function bridges the gap between graph mode and eager mode:

it allows Python functions like to_feature to work within TensorFlow's graph mode by converting TensorFlow tensors to NumPy tensors and back.

WRAP A PYTHON INTO A TENSOFLOW OP FOR EGAR EXCECUTION
Sentence A: "What is your name?"
Sentence B: "My name is John."
Features:
input_ids: Token IDs for [CLS] What is your name? [SEP] My name is John. [SEP].
input_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...] (to mask padding).
input_type_ids (or segment_ids):
[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, ...]:
0 for tokens in Sentence A.
1 for tokens in Sentence B.

The processed input features are organized into a dictionary:

Keys (input_word_ids, input_mask, input_type_ids) match what the BERT model expects.
Values are the respective tensors (input_ids, input_mask, segment_ids). 
The function returns:

x: A dictionary of input features ready for the BERT model.
label_id: The numeric label corresponding to the input text.

CREATING A TENSORFLOW PIPELINE WITH TF.DATA
This code prepares training (train_data) and validation (valid_data) datasets for a BERT-based model using TensorFlow’s tf.data
*Forces the data preprocessing operations to run on the CPU while model tranning is done in GPU
Pipeline for train_data
.shuffle(1000) - Randomly shuffles the data with a buffer size of 1000 to ensure the model doesn’t memorize the order of training data.
.batch(32, drop_remainder=True) - It’s like packing products into boxes. Each box (batch) holds 32 items. If there’s a leftover item that doesn’t fit into a box, it’s set aside (dropped).
.prefetch(tf.data.experimental.AUTOTUNE) - Prefetching is like preparing the next batch of products while the current batch is being shipped. It prevents idle time and improves efficiency.
--Pipeline for valid_data
For training, you shuffle questions to enhance learning. For validation, you keep the questions in a fixed order to fairly assess the student’s progress.

By batching, shuffling, and prefetching, the pipeline ensures that the data is efficiently fed into the model during training (keras.Model.fit)

--OutPUT 
Features:
input_mask, input_type_ids, and input_word_ids have shape (32, 128), meaning:
32 examples per batch.
Each example has a sequence length of 128 tokens.
Labels:
Shape (32,), meaning there is one label per example in a batch of 32.

ADD A CLASSIFICARION HEAD TO THE BERT LAYER 
This code defines a TensorFlow Keras model for a binary classification task using a BERT layer. 


Layer	Purpose	Real-Life Analogy
Inputs (input_word_ids, input_mask, input_type_ids)	- Ingredients for a recipe.
BERT Layer	-	Food processor transforming raw ingredients into a dish.
Dropout Layer	-	Taste-testing by skipping some ingredients.
Dense Layer	- Converts features into a probability for binary classification (0 or 1).	The final judgment of the dish (good/bad).
Model Creation - Combines the inputs, processing steps, and outputs into a cohesive model.	A structured kitchen workflow ready to produce dishes.

FINE - TUNE BERT FOR TEXT CLASSIFICATION
This section of code compiles, summarizes, and trains the BERT-based binary classification model.










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
