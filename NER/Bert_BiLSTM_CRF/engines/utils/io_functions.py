import json
import csv
import pandas as pd
lines = []
with open("/media/being/_dev_dva/KG/NLU/intent_recognition/Bert/data/test.csv", 'r', encoding='utf-8') as file_obj:
    for line in file_obj.readlines():
        line.replace('\n', '').replace(' ', '\t')
        lines.append('_!_'.join(line.rsplit(",", maxsplit=2)))

with open("/media/being/_dev_dva/KG/NLU/intent_recognition/Bert/data/test_file.csv", 'w', encoding='utf-8') as f:
    f.writelines(lines)

def read_csv(file_name, names, delimiter='t'):
    with open(file_name, 'r', encoding='utf-8') as file_obj:
        for line in file_obj.readlines():
            line.replace('\n', '').replace(' ', '\t')

    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, encoding='utf-8', skip_blank_lines=False, header=None, names=names)