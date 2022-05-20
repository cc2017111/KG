import math
import time
import warnings
import tensorflow as tf
from Bert_BiLSTM_CRF_Model import Bert_BiLSTM_CRF_MODEL
from tensorflow_addons.text.crf import crf_log_likelihood, crf_decode
from transformers import BertTokenizer
from utils.metrics import metrics


warnings.filterwarnings("ignore")
pretrain_model_name = "bert-base-chinese"
MODEL_PATH = "./bert-base-chinese"
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2"])
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)


def train(configs, dataManager, logger):
    vocab_size = dataManager.max_token_num
    num_classes = dataManager.max_label_num
    learning_rate = configs.learning_rate
    max_to_keep = configs.max_to_keep
    checkpoints_dir = configs.checkpoints_dir
    checkpoint_name = configs.checkpoint_name
    best_acc = 0.0
    best_at_epoch = 0
    unprocess = 0
    very_start_time = time.time()
    epoch = configs.epoch
    batch_size = configs.batch_size
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset, valid_dataset = dataManager.get_dataset()
    train_dataset = train_dataset.shuffle(len(train_dataset)).batch(global_batch_size)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    valid_dataset = valid_dataset.shuffle(len(valid_dataset)).batch(global_batch_size)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    with strategy.scope():

        model = Bert_BiLSTM_CRF_MODEL(bert_path=configs.bert_pretrain_path, configs=configs, num_classes=num_classes)

        if configs.optimizer == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate)
        elif configs.optimizer == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate)
        elif configs.optimizer == "RMSProp":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        elif configs.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate)

        # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        # test_loss = tf.keras.metrics.Mean(name='test_loss')
        # train_loss = tf.keras.metrics.Mean(name='train_loss')
        checkpoints = tf.train.Checkpoint(model=model)
        checkpoints_manager = tf.train.CheckpointManager(checkpoint=checkpoints, directory=checkpoints_dir,
                                                         max_to_keep=max_to_keep, checkpoint_name=checkpoint_name)

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:

            X_train_batch, y_train_batch, att_mask_train_batch = inputs
            input_length = tf.math.count_nonzero(X_train_batch, 1)
            logits, log_likelihood, transition_params = model.call(input_ids=X_train_batch, input_length=input_length, targets=y_train_batch)

            loss = -tf.reduce_mean(log_likelihood)
            # train_loss.update_state(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        batch_pred_sequence, _ = crf_decode(logits, transition_params, input_length)
        measures, _ = metrics(
            X_train_batch, y_train_batch, batch_pred_sequence, configs, dataManager, tokenizer)
        res_str = ''
        for k, v in measures.items():
            res_str += (k + ': %.3f ' % v)
        logger.info('training batch loss: %.5f, %s' % (loss, res_str))
        return loss

    @tf.function
    def test_step(inputs):
        X_val_batch, y_val_batch, att_mask_val_batch = inputs
        input_length_val = tf.math.count_nonzero(X_val_batch, 1)
        logits_val, log_likelihood_val, transition_params_val = model.call(input_ids=X_val_batch, input_length=input_length_val, targets=y_val_batch)
        loss = -tf.reduce_mean(log_likelihood_val)
        # test_loss.update_state(loss)
        batch_pred_sequence_val, _ = crf_decode(logits_val, transition_params_val, input_length_val)
        measures, _ = metrics(
            X_val_batch, y_val_batch, batch_pred_sequence_val, configs, dataManager, tokenizer)
        res_str = ''
        for k, v in measures.items():
            res_str += (k + ': %.3f ' % v)
        logger.info('validating batch loss: %.5f, %s' % (loss, res_str))
        dev_f1_avg = 0
        val_results = {}
        for measure in configs.measuring_metrics:
            val_results[measure] = 0
        for k, v in measures.items():
            val_results[k] += v
        for k, v in val_results.items():
            # val_results[k] /= num_val_iterations
            # val_res_str += (k + ': %.3f ' % val_results[k])
            if k == 'f1':
                dev_f1_avg = val_results[k]
        # test_accuracy.update_state(y_val_batch, batch_pred_sequence_val)
        return loss, dev_f1_avg

    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(inputs):
        return strategy.run(test_step, args=(inputs,))

    logger.info(('+' * 20) + 'training start' + ('+' * 20))
    for i in range(epoch):
        logger.info('epoch: {}/{}'.format(i + 1, epoch))
        checkpoint_path = checkpoints_manager.restore_or_initialize()
        if checkpoint_path is not None:
            logger.info("restore checkpoint at {}".format(checkpoint_path))
        start_time = time.time()
        total_loss = 0.0
        num_train_batches = 0
        num_val_batches = 0

        for X in iter(train_dist_dataset):
            train_loss = distributed_train_step(X)
            num_train_batches += 1

            if num_train_batches % configs.print_per_batch == 0 and num_train_batches != 0:
                logger.info(
                    'training batch: %5d, loss:%.5f' % (num_train_batches, train_loss))

        # validation
        logger.info('start evaluate engines...')
        val_f1 = 0.0
        for X_val in iter(valid_dist_dataset):
            test_loss, val_f1 = distributed_test_step(X_val)
            num_val_batches += 1

            if num_val_batches % configs.print_per_batch == 0 and num_val_batches != 0:
                logger.info(
                    'validating batch: %5d, loss:%.5f' % (num_val_batches, test_loss))

        time_span = (time.time() - start_time) / 60
        logger.info('time consumption: %.2f(min)' % time_span)
        # val_acc = test_accuracy.result()

        if val_f1 > best_acc:
            unprocess = 0
            best_acc = val_f1
            best_at_epoch = i + 1
            checkpoints_manager.save()
            tf.saved_model.save(model, configs.pb_model_sava_dir)
            logger.info('saved the new best model with acc: %.3f' % best_acc)
        else:
            unprocess += 1
        print('best acc:', best_acc)

        if configs.is_early_stop:
            if unprocess >= configs.patient:
                logger.info('early stopped, no process obtained with {} epoch'. format(configs.patient))
                logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
                logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return
        # train_accuracy.reset_state()
        # test_accuracy.reset_state()
        # test_loss.reset_state()
        # train_loss.reset_state()

    logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
