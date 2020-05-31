#importing dependencies
import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# load data
# loading data and opening our inout data in the form of a txt file
# Project Gutenburg is where the data can be found
file = open("Frankenstein.txt").read()

# tokenization
# standardization
# What is tokenization? Tokenization is the process of breaking a stream of text up into words phrases symbol or a meaningful
# elements
def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()
    # instantiating the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    #tokenizing the text into tokens
    tokens = tokenizer.tokenize(input)
    #filtering the stopwords using lambda
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return "".join(filtered)

# preprocess the input data
processed_inputs = tokenize_words(file)

# chars to numbers
# convert characters in our int to numbers
# we'll sort yhe list of the set of all characters that apear in outi/p test and then use the enumerate fn to get numbers that 
# represent the characters
# we'll then create a dictionary that stores the keys and values, or the characters and the numbers that represent them
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i, c in enumerate(chars))

# check if the words to chars or chars to num (?!) has worked?
# just so we get asn idea of whether our process of conveting words to characters has worked
# we print the length of our variables
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:", input_len)
print("Total vocab:", vocab_len)

# seq length
# we're defining how long we want an individual seqence here
# an individual sequence is a complete mapping of input as integers
seq_length = 100
x_data = []
y_data = []

# loop through the sequence
# here we're going through the entire list ofi/ps and converting the chars to numbers with a for loop
# this will create a bunch of sequences where each sequences starts with the next character in the i/p data
# beginning with the first character
for i in range(0, input_len - seq_length, 1):
    #define i/p and o/p sequences
    #i/p is the current character plus the desired sequence length
    in_seq = processed_inputs[i:i + seq_length]
    #out sequences is the initial character plus total sequence length
    out_seq = processed = processed_inputs[i + seq_length]
    # covertig the list of characters to integers based on the previous values and appendingg the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

# check to see how many total input sequences we have
n_patterns = len(x_data)
print("Total Patterns:", n_patterns)

# convert input sequence to np array that our network can use
x = numpy.reshape(x_data, (n_patterns, seq_length, 1))
x = x/float(vocab_len)

# one-hot encoding our label dat
y = np_utils.to_categorical(y_data)

# creating the model
# creating a sequential model
model = Sequential()
model.add(LSTM(256, input_shape = (x.shape[1], x.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = 'softmax'))

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# saving weigts
filepath = "model_weigts_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
desired_callbacks = [checkpoint]

#fit model and let it train
model.fit(x, y, epochs = 4, batch_size=256, callbacks = desired_callbacks)

# recompile model with the saved weigts
filename = "model_weigts_saved.hdf5"
model.load_weights(filename)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# output of the model back into characters
num_to_char = dict((i,c) for i,c in enumerate(chars))

# random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data(start)
print("Random Seed: ")
print("\"",''.join([num_to_char[value] for value in pattern]),"\"")

# generate the text
for i in range(1000):
    x = numpy.reshape(pattern, (1,len(pattern), 1))
    x = x/float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
