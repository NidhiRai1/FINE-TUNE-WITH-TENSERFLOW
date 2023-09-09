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
