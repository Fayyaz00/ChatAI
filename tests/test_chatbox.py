import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chatbox


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.predict.return_value = [[0.1, 0.8, 0.25]]
        self.bot = chatbox.Chatbot(
            intents={'intents': [{'tag': 'greeting', 'responses': ['Hello!']}]},
            words=['hello', 'cat'],
            classes=['other', 'greeting', 'threshold'],
            model=self.model,
            tokenize=str.split,
            lemmatize=lambda word: {'cats': 'cat'}.get(word, word),
            array=lambda value: value,
        )

    def test_normalizes_and_lemmatizes(self):
        self.assertEqual(self.bot.clean_up_sentence('HELLO Cats'), ['hello', 'cat'])

    def test_bag_of_words(self):
        self.assertEqual(self.bot.bag_of_words('Cats cats unknown'), [0, 1])

    def test_empty_and_unknown_inputs_skip_prediction(self):
        for sentence in ('', 'unknown words'):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.bot.respond(sentence), chatbox.FALLBACK_RESPONSE)
        self.model.predict.assert_not_called()

    def test_known_input_predicts_and_applies_threshold(self):
        self.assertEqual(self.bot.predict_class('HELLO'), [
            {'intent': 'greeting', 'probability': '0.8'}
        ])
        self.model.predict.assert_called_once_with([[1, 0]], verbose=0)
        self.assertEqual(self.bot.respond('HELLO'), 'Hello!')

    def test_no_confident_prediction_uses_fallback(self):
        self.model.predict.return_value = [[0.1, 0.2, 0.25]]
        self.assertEqual(self.bot.respond('hello'), chatbox.FALLBACK_RESPONSE)

    def test_unknown_intent_uses_fallback(self):
        self.assertEqual(chatbox.get_response([{'intent': 'missing'}], self.bot.intents),
                         chatbox.FALLBACK_RESPONSE)


class StartupTests(unittest.TestCase):
    def test_missing_artifacts_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(chatbox.StartupError, 'python training.py'):
                chatbox.load_chatbot(directory)

    def test_missing_nltk_data_is_actionable(self):
        import sys
        from types import ModuleType, SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            for name in ('words.pkl', 'classes.pkl', 'chatbotmodel.h5'):
                (Path(directory) / name).touch()
            nltk = ModuleType('nltk')
            nltk.word_tokenize = Mock(side_effect=LookupError('missing'))
            stem = ModuleType('nltk.stem')
            stem.WordNetLemmatizer = lambda: SimpleNamespace(lemmatize=lambda word: word)
            numpy = ModuleType('numpy')
            tensorflow = ModuleType('tensorflow')
            keras = ModuleType('tensorflow.keras')
            models = ModuleType('tensorflow.keras.models')
            models.load_model = Mock()
            modules = {'nltk': nltk, 'nltk.stem': stem, 'numpy': numpy,
                       'tensorflow': tensorflow, 'tensorflow.keras': keras,
                       'tensorflow.keras.models': models}
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(chatbox.StartupError, 'nltk.downloader'):
                    chatbox.load_chatbot(directory)
            models.load_model.assert_not_called()

    def test_main_reports_startup_error_without_traceback(self):
        stderr = io.StringIO()
        with patch.object(chatbox, 'load_chatbot', side_effect=chatbox.StartupError('run setup')):
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(chatbox.main(), 1)
        self.assertIn('run setup', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
