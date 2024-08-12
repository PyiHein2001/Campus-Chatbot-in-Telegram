import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from tensorflow.keras.models import load_model
import random

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the data
with open('UniChatBot.json') as file:
    data = json.load(file)

# Load the trained model
model = load_model('chatbot_model.h5')

# Load the tokenizer and label encoder
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Synonyms dictionary
synonyms = {
    "*": ["နည်းပညာ တက္ကသိုလ်", "ကျောင်း", "TUM", "နည်းပညာ တက္ကသိုလ် (မန္တလေး)", "တက္ကသိုလ်"]
}
def normalize_text(text, synonyms):
    for key, values in synonyms.items():
        for value in values:
            text = text.replace(value, key)
    return text


def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    # Normalize the sentence
    sentence = normalize_text(sentence, synonyms)
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, data)
    return res

if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            break
        response = chatbot_response(message)
        print(f"Bot: {response}")
