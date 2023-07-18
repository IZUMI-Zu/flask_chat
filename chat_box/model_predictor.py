import json
import tensorflow as tf

from chat_box.bahdanau_attention import BahdanauAttention
from chat_box.decoder import Decoder
from chat_box.encoder import Encoder

class ModelPredictor:
    def __init__(self):
        pass

    def tokenize(self, vocab):
        """
        Tokenizes the vocabulary file and returns the tokenizer.

        Args:
        - vocab (str): The path to the vocabulary file.

        Returns:
        - tokenizer: The tokenizer object.

        This method reads the vocabulary file, constructs the tokenizer, and returns it.
        The tokenizer is used to convert words to numbers and perform padding.
        """
        with open(vocab, 'r', encoding='utf-8') as f:
            tokenize_config = json.dumps(json.load(f), ensure_ascii=False)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
        return tokenizer

    def preprocess_sentence(self, sentence):
        """
        Preprocesses a sentence by adding start and end tokens.

        Args:
        - sentence (str): The input sentence.

        Returns:
        - sentence (str): The preprocessed sentence.

        This method adds the start and end tokens to the input sentence.
        """
        sentence = 'start ' + sentence + ' end'
        return sentence

    def predict(self, sentence):
        """
        Generates a response for the given input sentence.

        Args:
        - sentence (str): The input sentence.

        Returns:
        - result (str): The generated response.

        This method takes an input sentence, preprocesses it, and generates a response using the trained model.
        It loads the tokenizer and the pre-trained model, tokenizes the input sentence, and performs padding.
        Then, it passes the input through the encoder and initializes the decoder hidden state.
        The decoder starts with the start token and iteratively generates the output tokens until the end token is reached.
        The generated tokens are concatenated to form the response.
        """
        vocab_inp_size = 20000
        vocab_tar_size = 20000
        embedding_dim = 128
        units = 256
        batch_size = 1000

        encoder1 = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        decoder1 = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        optimizer1 = tf.keras.optimizers.Adam()
        checkpoint1 = tf.train.Checkpoint(optimizer=optimizer1, encoder=encoder1, decoder=decoder1)

        input_tokenizer = self.tokenize("model/inp.vocab")
        target_tokenizer = self.tokenize("model/tar.vocab")

        checkpoint_dir = 'model/model_data'
        checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir))

        sentence = self.preprocess_sentence(sentence)

        inputs = input_tokenizer.texts_to_sequences([sentence])[0]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=20, padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''

        hidden = [tf.zeros((1, 256))]
        enc_out, enc_hidden = encoder1(inputs, hidden)
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([target_tokenizer.word_index['start']], 0)

        for t in range(4):
            predictions, dec_hidden, attention_weights = decoder1(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy() + 1

            if target_tokenizer.index_word[predicted_id] == 'end':
                break
            result += str(target_tokenizer.index_word[predicted_id]) + ' '

            dec_input = tf.expand_dims([predicted_id], 0)
        return result
