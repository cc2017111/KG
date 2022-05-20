import os
import random
import jieba
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer
from collections import Counter
from utils.io_functions import read_csv

base_path = Path(__file__).resolve().parent.parent


class BertDataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.train_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.train_file
        if configs.dev_file is not None:
            self.dev_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.PADDING = '[PAD]'
        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.vocabs_dir = configs.vocabs_dir
        self.label2id_file = str(base_path) + '/' + configs.vocabs_dir + '/label2id'
        self.label2id, self.id2label = self.load_labels()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_token_num = len(self.tokenizer.get_vocab())
        self.max_label_num = len(self.label2id)

    def load_labels(self):
        if not os.path.isfile(self.label2id_file):
            self.logger.info("label2id file does not exists, creating...")
            return self.build_labels()
        with open(self.label2id_file, mode='r', encoding='utf-8') as file:
            rows = file.readlines()
            label2id = {}
            id2label = {}
            for row in rows:
                label = row.split('\t')[0]
                id = row.split('\t')[1].strip()
                label2id[label] = id
                id2label[id] = label

        return label2id, id2label

    def build_labels(self):
        df_train = read_csv(self.train_file, names=['text', 'label_class', 'label'], delimiter='_!_')
        labels = list(set(df_train['label_class'][df_train['label_class'].notnull()]))
        id2label = dict(zip(range(0, len(labels)), labels))
        label2id = dict(zip(labels, range(0, len(labels))))
        with open(self.label2id_file, mode='w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(str(id2label[idx]) + '\t' + str(idx) + '\n')

        return label2id, id2label

    def get_training_set(self, ratio=0.9):
        df_train = read_csv(self.train_file, names=['text', 'label_class', 'label'], delimiter='_!_')
        X, y, att_mask, token_type_ids = self.prepare(df_train)

        num_samples = len(X)
        if self.dev_file is not None:
            X_train = X
            y_train = y
            att_mask_train = att_mask
            token_type_ids_train = token_type_ids
            X_val, y_val, att_mask_val, token_type_ids_val = self.get_valid_set()
        else:
            X_train = X[:int(num_samples * ratio)]
            y_train = y[:int(num_samples * ratio)]
            X_val = X[int(num_samples * ratio):]
            y_val = y[int(num_samples * ratio):]
            att_mask_train = att_mask[:int(num_samples * ratio)]
            att_mask_val = att_mask[int(num_samples * ratio):]
            token_type_ids_train = token_type_ids[:int(num_samples * ratio)]
            token_type_ids_val = token_type_ids[int(num_samples * ratio):]
            self.logger.info("validation set does not exist, built...")
        self.logger.info("train set size: {}, validation set size: {}".format(len(X_train), len(X_val)))
        return X_train, y_train, att_mask_train, token_type_ids_train, X_val, y_val, att_mask_val, token_type_ids_val

    def get_valid_set(self):
        df_val = read_csv(self.dev_file, names=['text', 'label_class', 'label'], delimiter='_!_')
        X_val, y_val, att_mask_val, token_type_ids_val = self.prepare(df_val)
        return X_val, y_val, att_mask_val, token_type_ids_val

    def get_dataset(self):
        X_train, y_train, att_mask_train, token_type_ids_train, X_val, y_val, att_mask_val, token_type_ids_val = self.get_training_set(ratio=0.8)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, att_mask_train, token_type_ids_train))
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val, token_type_ids_val))
        return train_dataset, valid_dataset

    def prepare(self, df):
        self.logger.info("loading data...")
        X= []
        y = []
        att_mask = []
        token_type_ids = []
        for index, record in tqdm(df.iterrows()):
            sentence = record.text
            label = record.label_class
            if len(sentence) < self.max_sequence_length - 2:
                tmp_x = self.tokenizer.encode(sentence)
                tmp_att_mask = [1] * len(tmp_x)
                tmp_y = self.label2id[label]

                tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                token_type_ids_tmp = self.max_sequence_length * [0]
                X.append(tmp_x)
                y.append(tmp_y)
                att_mask.append(tmp_att_mask)
                token_type_ids.append(token_type_ids_tmp)
            else:
                tmp_x = self.tokenizer.encode(sentence)
                tmp_x = tmp_x[:self.max_sequence_length - 2]
                tmp_y = self.label2id[label]
                att_mask_tmp = [1] * self.max_sequence_length
                token_type_ids_tmp = [0] * self.max_sequence_length
                X.append(tmp_x)
                y.append(tmp_y)
                att_mask.append(att_mask_tmp)
                token_type_ids.append(token_type_ids_tmp)
        return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32), np.array(att_mask, dtype=np.int32), np.array(token_type_ids, dtype=np.int32)

    def next_batch(self, x, y, att_mask, token_type_ids, start_index):
        last_index = start_index + self.batch_size
        x_batch = list(x[start_index:min(last_index, len(x))])
        y_batch = list(y[start_index:min(last_index, len(x))])
        att_mask_batch = list(token_type_ids[start_index:min(last_index, len(x))])
        token_type_ids_batch = list(token_type_ids[start_index:min(last_index, len(x))])
        if last_index > len(x):
            left_size = last_index - len(x)
            for i in range(left_size):
                index = np.random.randint(len(x))
                x_batch.append(x[index])
                y_batch.append(y[index])
                att_mask_batch.append(att_mask[index])
                token_type_ids_batch.append(token_type_ids[index])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        att_mask_batch = np.array(att_mask_batch)
        token_type_ids_batch = np.array(token_type_ids_batch)

        return x_batch, y_batch, att_mask_batch, token_type_ids_batch

    def prepare_single_sentence(self, sentence):
        sentence = list(sentence)
        if len(sentence) <= self.max_sequence_length - 2:
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
            x += [0 for _ in range(self.max_sequence_length - len(x))]
            att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
        else:
            sentence = sentence[:self.max_sequence_length - 2]
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
        token_type_ids = self.max_sequence_length * [0]
        return np.array([x]), np.array([att_mask]), np.array([token_type_ids]), np.array([sentence])