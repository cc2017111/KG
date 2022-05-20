# -*- coding: utf-8 -*-
# @Time : 2021/6/4 8:53
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : model.py
# @Software: PyCharm
from abc import ABC
import copy
import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood, crf_decode
from transformers import TFBertModel, BertConfig


class Bert_BiLSTM_CRF_MODEL(tf.keras.Model, ABC):
    def __init__(self, bert_path, configs, num_classes):
        super(Bert_BiLSTM_CRF_MODEL, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = TFBertModel.from_pretrained(bert_path)
        self.bert.trainable = False
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout
        self.num_classes = num_classes
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(self.num_classes, self.num_classes)))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_ids'),
                                  tf.TensorSpec(shape=[None], dtype=tf.int64, name='input_length'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='targets')])
    def call(self, input_ids, input_length, targets):

        embedding_outputs = self.bert(input_ids)
        sequence_outputs = embedding_outputs[0]
        dropout_outputs = self.dropout(sequence_outputs)
        bilstm_outputs = self.bilstm(dropout_outputs)
        logits = self.dense(bilstm_outputs)
        tensor_targets = tf.convert_to_tensor(targets, dtype=tf.int32)
        log_likelihood, self.transition_params = crf_log_likelihood(logits, tensor_targets, input_length, transition_params=self.transition_params)
        # transition_params = self.transition_params
        return logits, log_likelihood, self.transition_params
