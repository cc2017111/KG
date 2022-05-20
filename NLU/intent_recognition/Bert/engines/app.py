import tensorflow as tf
from bert_model import Bert_TextCNN
import flask
import argparse
from data import BertDataManager
from gevent import pywsgi
import os
from pathlib import Path
from configure import Configure
from utils.logger import get_logger
base_path = Path(__file__).resolve().parent.parent
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class Predictor:
    def __init__(self, configs, data_manager, logger):
        self.dataManager = data_manager
        self.logger = logger
        self.configs = configs
        num_classes = self.dataManager.max_label_num
        self.label2id, self.id2label = data_manager.load_labels()
        self.label_list = list(self.label2id.keys())
        self.logger.info('loading model parameter')
        self.Bert_TextCNN_Model = Bert_TextCNN(bert_path=os.path.join(base_path, 'engines/bert-base-chinese'), configs=configs, num_classes=num_classes)
        # print(configs.pb_model_sava_dir)
        # self.Bert_TextCNN_Model = tf.keras.models.load_model("/media/being/_dev_dva/KG/NLU/intent_recognition/Bert/engines/saved_model")
        checkpoints = tf.train.Checkpoint(model=self.Bert_TextCNN_Model)
        checkpoints_manager = tf.train.CheckpointManager(checkpoint=checkpoints, directory=configs.checkpoint_dir,
                                                         max_to_keep=configs.max_to_keep, checkpoint_name=configs.checkpoint_name)

        checkpoint_path = checkpoints_manager.restore_or_initialize()
        print(checkpoint_path)
        self.logger.info('loading model successfully...')

    def predict_one(self, sentence):
        x, att_mask, token_type_ids, sentence_ = self.dataManager.prepare_single_sentence(sentence)
        # print(x)
        logits = self.Bert_TextCNN_Model.call(input_ids=x, input_mask=att_mask, token_ids=token_type_ids, training=False).numpy()
        rst = {label:prob for label, prob in zip(self.label_list, logits[0])}
        rst = sorted(rst.items(), key=lambda kv:kv[1], reverse=True)
        print(rst)

        label, confidence = rst[0]
        return {"name": label, "confidence": str(confidence)}


parser = argparse.ArgumentParser(description='Bert_TextCNN')
parser.add_argument('--config_file', default='Bert_textCNN.config')
args = parser.parse_args()
configs = Configure(config_file=os.path.join(base_path, args.config_file))
logger = get_logger(configs.log_dir)
dataManager = BertDataManager(configs, logger)
predictor = Predictor(configs=configs, data_manager=dataManager, logger=logger)
# print(predictor.predict_one("岩骨斜坡脑膜瘤可以怎么预防"))

if __name__ == "__main__":
    app = flask.Flask(__name__)

    @app.route("/service/api/bert_intent_recognize", methods=["GET", "POST"])
    def bert_intent_recognize():
        data = {"success": 0}
        param = flask.request.get_json()
        print(param)
        text = param["text"]
        result = predictor.predict_one(text)
        data["data"] = result
        data["success"] = 1
        print("data", data)
        return flask.jsonify(data)

    server = pywsgi.WSGIServer(("0.0.0.0", 60062), app)
    server.serve_forever()

