import io
import random
import string # to process standard python strings
import warnings
import numpy as np
import os
import speech_recognition as sr
from gtts import gTTS
import playsound
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from englisttohindi.englisttohindi import EngtoHindi
from googletrans import Translator
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only

translator = Translator()

#Reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

num=1

def speak(output):
    global num

    # num to rename every audio file
    # with different name to remove ambiguity
    num += 1
    print("Jarvis : ", output)

    toSpeak = gTTS(text=output, lang='en', slow=False)
    # saving the audio file given by google text to speech
    file = str(num) + ".mp3"
    toSpeak.save(file)

    # playsound package is used to play the same file.
    playsound.playsound(file, True)
    os.remove(file)


def listening():
    reg = sr.Recognizer()
    audio = " "
    # intializing audio with an empty node
    with sr.Microphone() as source:
        print("listening...")
        audio = reg.listen(source, phrase_time_limit=3)
    print("Recognising")  # time limit is 5 sec as pause

    try:
        text = reg.recognize_google(audio, language="en-in")
        print("you:", text)
        return (text)

    except:
        speak(EngtoHindi("sorry sir i couldn't get this").convert)
        return "sorry sir i couldn't get this";

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        return 'can you please repeat'
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response.replace(user_response,'',1)


flag=True
# print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
speak(EngtoHindi("Hello sir/maam,This is regarding dues of credit card bills").convert)
# speak("Hello sir/maam,This is regarding dues of credit card bills")
# speak("Can you please tell us the reason for the dues")
speak(EngtoHindi("Can you please tell us the reason for the dues").convert)

while(flag==True):
    # user_response = input()
    user_response1=listening().lower()
    user_response=translator.translate(user_response1, src='hi',dest='en').text
    # user_response=listening().lower()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False

            speak("You are Welcome")
        else:

            if(greeting(user_response)!=None):

                speak(greeting(user_response))
            else:

                # speak(response(user_response))
                speak(EngtoHindi(response(user_response)).convert)
                sent_tokens.remove(user_response)
    else:
        flag=False
        # print("ROBO: Bye! take care..")
        speak(EngtoHindi("If your issue is not been solved please contact our customer care number 1800-222-333 ").convert)
        speak(EngtoHindi("Ok sir/maam,please take care of this because it will ruin your credit score").convert)
        speak(EngtoHindi("Bye,Take care...").convert)