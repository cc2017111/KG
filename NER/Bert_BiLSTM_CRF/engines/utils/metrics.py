# -*- coding: utf-8 -*-
# @Time : 2021/6/4 8:53
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : metrics.py
# @Software: PyCharm
import re



def metrics(X, y_true, y_pred, configs, data_manager, tokenizer):
    precision = -1.0
    recall = -1.0
    f1 = -1.0

    hit_num = 0
    pred_num = 0
    true_num = 0

    correct_label_num = 0
    total_label_num = 0

    label_num = {}
    label_metrics = {}
    measuring_metrics = configs.measuring_metrics
    # tensor向量不能直接索引，需要转成numpy
    y_pred = y_pred.numpy()
    for i in range(len(y_true)):
        if configs.use_bert:
            x = tokenizer.convert_ids_to_tokens(X[i].tolist(), skip_special_tokens=True)
        else:
            x = [str(data_manager.id2token[val]) for val in X[i] if val != data_manager.token2id[data_manager.PADDING]]
        y = [str(data_manager.id2label[val]) for val in y_true[i] if val != data_manager.label2id[data_manager.PADDING]]
        y_hat = [str(data_manager.id2label[val]) for val in y_pred[i] if
                 val != data_manager.label2id[data_manager.PADDING]]  # if val != 5

        correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
        total_label_num += len(y)
        print(x, y)
        true_labels, labeled_labels_true, _ = extract_entity(x, y, data_manager)
        print(true_labels, labeled_labels_true)
        print(x, y_hat)
        pred_labels, labeled_labels_pred, _ = extract_entity(x, y_hat, data_manager)
        print(pred_labels, labeled_labels_pred)

        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))
        
        for label in data_manager.suffix:
            label_num.setdefault(label, {})
            label_num[label].setdefault('hit_num', 0)
            label_num[label].setdefault('pred_num', 0)
            label_num[label].setdefault('true_num', 0)

            true_lab = [x for (x, y) in zip(true_labels, labeled_labels_true) if y == label]
            pred_lab = [x for (x, y) in zip(pred_labels, labeled_labels_pred) if y == label]

            label_num[label]['hit_num'] += len(set(true_lab) & set(pred_lab))
            label_num[label]['pred_num'] += len(set(pred_lab))
            label_num[label]['true_num'] += len(set(true_lab))

    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    # 按照字段切分
    for label in label_num.keys():
        tmp_precision = 0
        tmp_recall = 0
        tmp_f1 = 0
        # 只包括BI
        if label_num[label]['pred_num'] != 0:
            tmp_precision = 1.0 * label_num[label]['hit_num'] / label_num[label]['pred_num']
        if label_num[label]['true_num'] != 0:
            tmp_recall = 1.0 * label_num[label]['hit_num'] / label_num[label]['true_num']
        if tmp_precision > 0 and tmp_recall > 0:
            tmp_f1 = 2.0 * (tmp_precision * tmp_recall) / (tmp_precision + tmp_recall)
        label_metrics.setdefault(label, {})
        label_metrics[label]['precision'] = tmp_precision
        label_metrics[label]['recall'] = tmp_recall
        label_metrics[label]['f1'] = tmp_f1

    results = {}
    for measure in measuring_metrics:
        results[measure] = vars()[measure]
    return results, label_metrics

def extract_entity_(sentence, labels_, reg_str, label_level):
    entices = []
    labeled_labels = []
    labeled_indices = []
    labels__ = [('%03d' % ind) + lb for lb, ind in zip(labels_, range(len(labels_)))]
    labels = ' '.join(labels__)

    re_entity = re.compile(reg_str)

    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        if label_level == 1:
            labeled_labels.append('_')
        elif label_level == 2:
            labeled_labels.append(entity_labels.split()[0][5:])

        start_index = int(entity_labels.split()[0][:3])
        if len(entity_labels.split()) != 1:
            end_index = int(entity_labels.split()[-1][:3]) + 1
        else:
            end_index = start_index + 1
        entity = ' '.join(sentence[start_index-1:end_index-1])
        labels = labels__[end_index:]
        labels = ' '.join(labels)
        entices.append(entity)
        labeled_indices.append((start_index, end_index))
        m = re_entity.search(labels)

    return entices, labeled_labels, labeled_indices


def extract_entity(x, y, data_manager):
    label_scheme = data_manager.label_scheme
    label_level = data_manager.label_level
    label_hyphen = data_manager.hyphen
    reg_str = ''
    if label_scheme == 'BIO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r'\s*)*'

    elif label_scheme == 'BIESO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*([0-9][0-9][0-9]E' + r' )|([0-9][0-9][0-9]S' + r' )'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r' )*([0-9][0-9][0-9]E' + label_hyphen + tag_str + r' )|([0-9][0-9][0-9]S' + label_hyphen + tag_str + r' )'

    return extract_entity_(x, y, reg_str, label_level)
