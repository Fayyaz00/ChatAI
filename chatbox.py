import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path


FALLBACK_RESPONSE = "I'm not sure how to respond to that."
ERROR_THRESHOLD = 0.25


class StartupError(Exception):
    """An actionable problem that prevents the chatbot from starting."""


@dataclass
class Chatbot:
    intents: dict
    words: list
    classes: list
    model: object
    tokenize: object
    lemmatize: object
    array: object

    def clean_up_sentence(self, sentence):
        return [self.lemmatize(word.lower()) for word in self.tokenize(sentence)]

    def bag_of_words(self, sentence):
        sentence_words = set(self.clean_up_sentence(sentence))
        return [int(word in sentence_words) for word in self.words]

    def predict_class(self, sentence):
        bag = self.bag_of_words(sentence)
        if not any(bag):
            return []
        probabilities = self.model.predict(self.array([bag]), verbose=0)[0]
        results = [
            {"intent": self.classes[index], "probability": str(probability)}
            for index, probability in sorted(
                enumerate(probabilities), key=lambda item: item[1], reverse=True
            )
            if probability > ERROR_THRESHOLD
        ]
        return results

    def respond(self, sentence):
        return get_response(self.predict_class(sentence), self.intents)


def get_response(intents_list, intents_json):
    if not intents_list:
        return FALLBACK_RESPONSE
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return FALLBACK_RESPONSE


def load_chatbot(directory=Path('.')):
    """Load the trained artifacts and check dependencies before starting chat."""
    directory = Path(directory)
    artifact_names = ('words.pkl', 'classes.pkl', 'chatbotmodel.h5')
    missing = [name for name in artifact_names if not (directory / name).is_file()]
    if missing:
        raise StartupError(
            f"Missing model files: {', '.join(missing)}. "
            "Run 'python training.py' from the repository directory."
        )
    try:
        import nltk
        import numpy as np
        from nltk.stem import WordNetLemmatizer
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise StartupError(
            "Could not load a required dependency. Run 'python -m pip install -r "
            "requirements.txt' and check that TensorFlow can run on this system. "
            f"Details: {exc}"
        ) from exc

    lemmatizer = WordNetLemmatizer()
    try:
        nltk.word_tokenize('test')
        lemmatizer.lemmatize('tests')
    except LookupError as exc:
        raise StartupError(
            "Missing NLTK language data. Run 'python -m nltk.downloader "
            "punkt punkt_tab wordnet'."
        ) from exc

    try:
        with (directory / 'intents.json').open(encoding='utf-8') as file:
            intents = json.load(file)
        with (directory / 'words.pkl').open('rb') as file:
            words = pickle.load(file)
        with (directory / 'classes.pkl').open('rb') as file:
            classes = pickle.load(file)
        model = load_model(directory / 'chatbotmodel.h5')
    except (OSError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        raise StartupError(
            f"Could not load chatbot data: {exc}. Check intents.json and run "
            "'python training.py' to regenerate the model files."
        ) from exc
    return Chatbot(intents, words, classes, model, nltk.word_tokenize,
                   lemmatizer.lemmatize, np.array)


def main():
    try:
        chatbot = load_chatbot()
    except StartupError as exc:
        print(f'Cannot start chatbot: {exc}', file=sys.stderr)
        return 1
    print('Go! Bot is running! Press Ctrl+C to exit.')
    try:
        while True:
            print(chatbot.respond(input('')))
    except (EOFError, KeyboardInterrupt):
        print('\nGoodbye!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
