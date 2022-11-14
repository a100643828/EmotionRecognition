
import jieba.posseg as pseg
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.preprocessing import text
from tensorflow import keras
import numpy as np
import pickle
import pandas as pd


class NLP(object):
    MAX_SEQUENCE_LENGTH = 200
    MAX_NUM_WORDS = 10000
    label_to_index = {
        '正面': 1,
        '負面': 0
    }
    new_model = tf.keras.models.load_model('ScoreEmotionRecognition\Model')
    tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)

    def __init__(self):
        with open('ScoreEmotionRecognition\\tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def jieba_tokenizer(self,text):
        words = pseg.cut(text)
        return ' '.join([
            word for word, flag in words if flag != 'x'])

    
    def EmotionRecognition(self, text):
        cols = ['text', 'title1_tokenized', 'label']
        df = pd.DataFrame(columns=cols)
        text_input_jeiba = self.jieba_tokenizer(text)
        df.loc[0]=[text, text_input_jeiba, '']
        x1_test=self.tokenizer.texts_to_sequences(df.title1_tokenized)
        x1_test=keras.preprocessing.sequence.pad_sequences(
            x1_test, maxlen = self.MAX_SEQUENCE_LENGTH)
        predictions=self.new_model.predict(x1_test)
        index_to_label={v: k for k, v in self.label_to_index.items()}
        df['label']=[index_to_label[idx] for idx in np.argmax(predictions, axis= 1)]
        return df.iloc[0]['label']

