#save in .txt
import logging
import httpx
from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from seg import segment_word
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the data and model
with open('UniChatBot_segmented.json') as file:
    data = json.load(file)

model = load_model('chatbot_model.h5')

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    metrics=['accuracy']
)

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Define synonyms
synonyms = {
    "ကျောင်း": ["နည်းပညာ တက္ကသိုလ်", "ကျောင်း", "TUM", "နည်းပညာ တက္ကသိုလ် (မန္တလေး)", "တက္ကသိုလ်"],
    "လား": ["သလား"],
    "Civil Engineering": ["civil", "CIVIL", "Civil"],
    "Electronic Engineering": ["EC", "Ec", "ec"],
    "Electrical Power Engineering": ["EP", "Ep", "ep"],
    "Information Technology": ["IT", "It", "it"],
    "Mechanical Engineering": ["ME", "Me", "me"],
    "Mechatronic Engineering": ["MC", "Mc", "mc"],
}

def normalize_text(text, synonyms):
    """Replace synonyms in the text with the corresponding key."""
    for key, values in synonyms.items():
        for value in values:
            text = text.replace(value, key)
    return text

def myanmar_tokenize(text):
    """Tokenize Myanmar text."""
    return segment_word(text).split()

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence."""
    sentence_words = myanmar_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Convert a sentence into a bag of words representation."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    """Predict the class of the sentence using the model."""
    sentence = normalize_text(sentence, synonyms)
    print(f"Normalized sentence: {sentence}")  # Debug
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"Results: {results}")  # Debug
    return_list = []
    if results:
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    else:
        return_list.append({"intent": "noanswer", "probability": "1.0"})
    print(f"Return list: {return_list}")  # Debug
    return return_list

def get_response(ints, intents_json):
    """Generate a response based on the predicted intent."""
    print(f"Intents: {ints}")  # Debug
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            print(f"Response: {result}")  # Debug
            return result
    # If no intent matches, return the "noanswer" response
    for i in list_of_intents:
        if i['tag'] == "noanswer":
            result = random.choice(i['responses'])
            print(f"Noanswer Response: {result}")  # Debug
            return result
    return "မေးခွန်းတွက် သင့်တော်သော အဖြေ မပေးနိုင် ၍ ဝမ်းနည်း ပါတယ်။ ကျောင်းသားရေးရာ ဖုန်းနံပါတ် ၀၉-၉၈၈၄၈၄၁၇၂ ကို ဆက်သွယ်၍ အသေးစိတ်ကို မေးမြန်း နိုင်ပါတယ်။ "

def chatbot_response(text):
    """Generate a response from the chatbot."""
    print(f"User input: {text}")  # Debug
    ints = predict_class(text, model)
    res = get_response(ints, data)
    if not res:
        res = "မေးခွန်းတွက် သင့်တော်သော အဖြေ မပေးနိုင် ၍ ဝမ်းနည်း ပါတယ်။ ကျောင်းသားရေးရာ ဖုန်းနံပါတ် ၀၉-၉၈၈၄၈၄၁၇၂ ကို ဆက်သွယ်၍ အသေးစိတ်ကို မေးမြန်း နိုင်ပါတယ်။ "
    return res, ints[0]['intent']

def log_user_input(user_id, user_message, predicted_intent):
    """Log user inputs to a file."""
    with open("user_inputs.txt", "a") as f:
        f.write(f"User ID: {user_id}, Message: {user_message}, Predicted Intent: {predicted_intent}\n")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the /start command is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"မင်္ဂလာပါ {user.mention_html()}! ကျွန်တော်က တော့ BotTUM ဖြစ်ပါတယ်။ TUM နဲ့ပါတ်သတ်၍ Information တွေကို ဖြေကြား ပေးနိုင်ပါတယ်။",
        reply_markup=ForceReply(selective=True),
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    user_message = update.message.text
    user_id = update.message.from_user.id
    bot_response, predicted_intent = chatbot_response(user_message)
    log_user_input(user_id, user_message, predicted_intent)
    await update.message.reply_text(bot_response)

def main() -> None:
    """Start the bot."""
    TOKEN = '7286467450:AAHuvqOC9jiosqYRi_fTCJzJwbbJgAZQXlg'

    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Create the application
    application = Application.builder().token(TOKEN).build()

    # Set timeout settings
    application.bot.request.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=15.0))

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == '__main__':
    main()
