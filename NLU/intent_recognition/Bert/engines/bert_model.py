import tensorflow as tf
from transformers import BertConfig, TFBertModel


class Bert_TextCNN(tf.keras.Model):
    def __init__(self, bert_path, configs, num_classes):
        super(Bert_TextCNN, self).__init__()
        self.num_filters = configs.num_filters
        self.num_classes = num_classes
        self.embedding_dim = configs.embedding_dim
        self.sequence_length = configs.max_sequence_length
        self.dropout_rate = configs.dropout_rate
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert_layers = TFBertModel.from_pretrained(bert_path, num_labels=self.num_classes)
        self.bert_layers.trainable = True
        self.softmax = tf.keras.layers.Softmax()
        self.conv1 = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=3,
                                            strides=1,
                                            padding='same', activation='relu')
        self.pool1 = tf.keras.layers.GlobalMaxPool1D()
        self.conv2 = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=4,
                                            strides=1,
                                            padding='same', activation='relu')
        self.pool2 = tf.keras.layers.GlobalMaxPool1D()
        self.conv3 = tf.keras.layers.Conv1D(filters=self.num_filters, kernel_size=5,
                                            strides=1,
                                            padding='same', activation='relu')
        self.pool3 = tf.keras.layers.GlobalMaxPool1D()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')
        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                           bias_regularizer=tf.keras.regularizers.l2(0.2), name='dense1')
        self.dense2 = tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                           bias_regularizer=tf.keras.regularizers.l2(0.2), name='dense2')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_ids'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_mask'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='token_ids'),
                                  tf.TensorSpec(shape=[], dtype=tf.bool, name='training')])
    def call(self, input_ids, input_mask, token_ids, training=True):
        embedding_outputs = self.bert_layers(inputs=input_ids, attention_mask=input_mask, token_type_ids=token_ids)
        # TextCNN_feature
        inputs = embedding_outputs[0]
        # print("inputs:", inputs)
        pooled_output = []
        con1 = self.conv1(inputs)
        # print("con1", con1)
        pool1 = self.pool1(con1)
        # print("pool1", pool1)
        pooled_output.append(pool1)

        con2 = self.conv2(inputs)
        pool2 = self.pool2(con2)
        pooled_output.append(pool2)

        con3 = self.conv3(inputs)
        pool3 = self.pool3(con3)
        pooled_output.append(pool3)

        concat_outputs = tf.keras.layers.concatenate(pooled_output, axis=-1, name='concatenate1')
        cnn_feature = self.dropout(concat_outputs, training)

        all_concat_feature = tf.keras.layers.concatenate([cnn_feature, embedding_outputs[1]], axis=-1, name='concatenate2')
        outputs = self.dense1(all_concat_feature)
        outputs = self.dense2(outputs)
        return outputs
