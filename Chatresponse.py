import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tflearn
import tensorflow as tf
import random
import requests
import speech_recognition as sr
import spacy
nlp = spacy.load("en")




# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))




# load our saved model
model.load('./model.tflearn')



# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def extract_entity_names(t):
    entity_names = []
    
    if hasattr(t, 'node'):
        if t.node == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
       
    return entity_names

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            if results[0][0]=="weather":
                sentence = sentence.decode("UTF-8")
                sentence = nlp(sentence)
                place = ""
                for entity in sentence.ents:
                    place = entity.text
                response = requests.get("http://api.openweathermap.org/data/2.5/weather?q="+place+"&appid=885bbeb8ae784829205572c9adb2c5f6")
                current = response.content
                current = json.loads(current)
                current = current['weather']
                current = current[0]
                current = current['description']

                print "It seems now "+place+" has "+current
                results.pop(0)
                continue


            for i in intents['intents']:
                # find a tag matching the first result        
                if i['tag'] == results[0][0]:
                
                    print  random.choice(i['responses'])

            results.pop(0)


print "Hi , I am tiny , the bot , How do you want me to help?"
    #Convo starts 

r = sr.Recognizer()
while(True):
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        req = r.recognize_google(audio)
        req = req.encode("UTF-8")
        #print "************************************************"
        #print req
        response(req)
    
     
    except sr.UnknownValueError:
        print("Don't you want to say something???Sorry I don't recognize..")
    