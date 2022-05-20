import json
import ahocorasick
import tensorflow as tf
from Bert_BiLSTM_CRF_Model import Bert_BiLSTM_CRF_MODEL
from tensorflow_addons.text.crf import crf_decode
import argparse
from gevent import pywsgi
from data import BertDataManager
import os
import flask
from pathlib import Path
from configure import Configure
from utils.logger import get_logger
from utils.extract_entity import extract_entity
base_path = Path(__file__).resolve().parent.parent
print("base_path:", base_path)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class NerBaseDict(object):
    def __init__(self, dict_path):
        super(NerBaseDict, self).__init__()
        self.dict_path = dict_path
        self.region_words = self.load_dict(self.dict_path)
        self.region_tree = self.build_actree(self.region_words)

    def load_dict(self, path):
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)

    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    def recognize(self, text):
        item = {"String": text, "entities": []}
        region_wds = []
        for i in self.region_tree.iter(text):
            wd = i[1][1]
            region_wds.append(wd)
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        item["entities"] = [{"word": i, "type": "disease", "recog_label": "dict"} for i in final_wds]
        return item

class Predictor:
    def __init__(self, configs, data_manager, logger):
        self.dataManager = data_manager
        self.logger = logger
        self.configs = configs
        num_classes = self.dataManager.max_label_num
        self.label2id, self.id2label = data_manager.load_labels()
        self.label_list = list(self.label2id.keys())
        self.logger.info('loading model parameter')
        self.Bert_BiLSTM_CRF_MODEL = Bert_BiLSTM_CRF_MODEL(bert_path='bert-base-chinese', configs=configs, num_classes=num_classes)
        self.nbd = NerBaseDict(os.path.join(base_path, "data/diseases.json"))
        # print(configs.pb_model_sava_dir)
        # self.Bert_TextCNN_Model = tf.keras.models.load_model("/media/being/_dev_dva/KG/NLU/intent_recognition/Bert/engines/saved_model")

        checkpoint = tf.train.Checkpoint(model=self.Bert_BiLSTM_CRF_MODEL)
        status = checkpoint.restore(tf.train.latest_checkpoint(os.path.join(base_path, "engines/checkpoints")))
        print(os.path.join(base_path, "engines/checkpoints"), status)
        self.logger.info('loading model successfully...')

    def predict_one(self, sentence):
        x, y, att_mask, sentence_ = self.dataManager.prepare_single_sentence(sentence)
        input_length = tf.math.count_nonzero(x, 1)
        logits, log_likelihood, transition_params = self.Bert_BiLSTM_CRF_MODEL.call(input_ids=x, input_length=input_length, targets=y)
        # print("transition_params", transition_params)
        label_predicts, _ = crf_decode(logits, transition_params, input_length)
        label_predicts = label_predicts.numpy()
        sentence_temp = sentence_[0, 0:input_length[0]]
        # print("sentence_", sentence_)
        # print("sentence_temp:", sentence_temp)
        # print("label_predicts:", label_predicts)
        # y_pred = [str(self.dataManager.id2label[val]) for val in label_predicts[0][0:input_length[0]]]
        y_pred = [str(self.dataManager.id2label[val]) for val in label_predicts[0][0:input_length[0]] if val != self.dataManager.label2id[self.dataManager.PADDING]]
        y_pred = y_pred[1:-1]
        # print("y_pred1:", y_pred)
        entities, suffixes, indices = extract_entity(sentence_temp, y_pred, self.dataManager)
        ents1 = {"String": sentence, "entities": []}
        ents1["entities"] = [{"word": entities[i], "type": "disease", "recog_label": "model"} for i in range(len(entities))]
        res = []
        if ents1["entities"]:
            res.append(ents1)
        print(ents1)
        ents2 = self.nbd.recognize(sentence)
        if ents2["entities"]:
            res.append(ents2)

        return res

parser = argparse.ArgumentParser(description='Bert_BiLSTM_CRF')
parser.add_argument('--config_file', default='Bert_BiLSTM_CRF.config')
args = parser.parse_args()
configs = Configure(config_file=os.path.join(base_path, args.config_file))
logger = get_logger(configs.log_dir)
dataManager = BertDataManager(configs, logger)
predictor = Predictor(configs=configs, data_manager=dataManager, logger=logger)
# print(predictor.predict_one("岩骨斜坡脑膜瘤可以怎么预防"))

if __name__ == "__main__":
    app = flask.Flask(__name__)
    @app.route("/service/api/medical_ner", methods=["GET", "POST"])
    def medical_ner():
        data = {"success": 0}
        result = []
        text_list = flask.request.get_json()["text_list"]
        result = predictor.predict_one(text_list)
        data["data"] = result
        data["success"] = 1
        return flask.jsonify(data)
    server = pywsgi.WSGIServer(("0.0.0.0", 60061), app)
    server.serve_forever()

    # print(predictor.predict_one("上海治疗性病的专业医院查询性病患者，患病两个月了，一直未好，得的是尖锐湿疣。请问性病好治不？怎么治疗好？"))
