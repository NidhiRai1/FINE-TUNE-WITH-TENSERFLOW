# FINE-TUNE-WITH-TENSERFLOW

//NLP stands for Natural Language Processing. It gives the ability to machine understand and process human languages. Human languages can be in the form of text or audio format.
----advantage - NLP helps us to analyse data from both structured and unstructured sources.
NLP is very fast and time efficient.
----disadvantage - NLP  require huge data 
-----APPLICATION - text classification , audio processing 

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
