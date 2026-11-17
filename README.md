# ChatAI

A simple intent-based chatbot built with NLTK and TensorFlow. It trains on the patterns and responses in `intents.json`.

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab wordnet
```

Train the model and start the chatbot from the repository directory:

```sh
python training.py
python chatbox.py
```

Training creates `words.pkl`, `classes.pkl`, and `chatbotmodel.h5` locally. Retrain after changing `intents.json`. Press Ctrl+C (or send EOF) to exit the chat.

## Tests

Run the automated tests without trained model files or TensorFlow:

```sh
python -m unittest discover -s tests -v
```

A full smoke test requires a working TensorFlow installation and the NLTK data above. Run `python training.py`, then `python chatbox.py`; try a greeting, an unknown phrase, and Ctrl+C or EOF to exit.
