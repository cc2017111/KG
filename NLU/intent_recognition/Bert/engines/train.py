import math
import time
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from bert_model import Bert_TextCNN
from transformers import BertTokenizer


warnings.filterwarnings("ignore")
pretrain_model_name = "bert-base-chinese"
MODEL_PATH = "./bert-base-chinese"
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2"])
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)

def train(configs, dataManager, logger):
    label_list = ["传染性", "预防","治疗方法","其他","治疗时间","相关病症","临床表现(病症表现)","定义","禁忌","所属科室","治愈率","化验/体检方案","病因"]
    vocab_size = dataManager.max_token_num
    num_classes = dataManager.max_label_num
    learning_rate = configs.learning_rate
    max_to_keep = configs.max_to_keep
    checkpoint_dir = configs.checkpoint_dir
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

        model = Bert_TextCNN(bert_path=configs.bert_pretrain_path, configs=configs, num_classes=num_classes)

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

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        checkpoints = tf.train.Checkpoint(model=model)
        checkpoints_manager = tf.train.CheckpointManager(checkpoint=checkpoints, directory=checkpoint_dir,
                                                         max_to_keep=max_to_keep, checkpoint_name=checkpoint_name)

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:

            X_train_batch, y_train_batch, att_mask_train_batch, token_type_ids_train_batch = inputs
            # print(X_train_batch, y_train_batch, att_mask_train_batch, token_type_ids_train_batch)
            outputs = model.call(input_ids=X_train_batch, input_mask=att_mask_train_batch, token_ids=token_type_ids_train_batch, training=True)
            y_true = tf.one_hot(y_train_batch, depth=num_classes)

            losses = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=outputs, from_logits=False)
            loss = tf.reduce_mean(losses)
            train_loss.update_state(losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(y_train_batch, outputs)
        return loss, outputs

    @tf.function
    def test_step(inputs):
        X_val_batch, y_val_batch, att_mask_val_batch, token_type_ids_val_batch = inputs
        outputs = model.call(input_ids=X_val_batch, input_mask=att_mask_val_batch, token_ids=token_type_ids_val_batch, training=False)
        y_true = tf.one_hot(y_val_batch, depth=num_classes)
        losses = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=outputs, from_logits=False)
        test_loss.update_state(losses)
        test_accuracy.update_state(y_val_batch, outputs)
        return losses, outputs

    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses, outputs = strategy.run(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), outputs

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
            X_train_batch, y_train_batch, att_mask_train_batch, token_type_ids_train_batch = X
            y_train_batch = y_train_batch.values[0].numpy()
            X_train_batch = X_train_batch.values[0].numpy()[0]
            sentence = tokenizer.decode(X_train_batch)
            y_true = tf.one_hot(y_train_batch, depth=num_classes)
            # print(X_train_batch, y_train_batch)
            _, outputs = distributed_train_step(X)
            rst = {label: prob for label, prob in zip(label_list, outputs.values[0].numpy()[0])}
            rst = sorted(rst.items(), key=lambda kv: kv[1], reverse=True)
            print(rst[0], label_list[np.argmax(y_true[0])], sentence)
            num_train_batches += 1

            if num_train_batches % configs.print_per_batch == 0 and num_train_batches != 0:
                logger.info(
                    'training batch: %5d, loss:%.5f, acc:%.5f' % (num_train_batches, train_loss.result(), train_accuracy.result()))

        # validation
        logger.info('start evaluate engines...')

        for X_val in iter(valid_dist_dataset):
            _ = distributed_test_step(X_val)
            num_val_batches += 1

            if num_val_batches % configs.print_per_batch == 0 and num_val_batches != 0:
                logger.info(
                    'validating batch: %5d, loss:%.5f, acc:%.5f' % (num_val_batches, test_loss.result(), test_accuracy.result()))

        time_span = (time.time() - start_time) / 60
        logger.info('time consumption: %.2f(min)' % time_span)
        val_acc = test_accuracy.result()

        if val_acc > best_acc:
            unprocess = 0
            best_acc = val_acc
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
        train_accuracy.reset_state()
        test_accuracy.reset_state()
        test_loss.reset_state()
        train_loss.reset_state()

    logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
