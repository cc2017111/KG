import os
import json
from pathlib import Path
base_path = Path(__file__).resolve().parent
LOGS_DIR = os.path.join(base_path, 'log')
def dump_user_dialogue_context(user, data):
    print("user", str(user))
    path = os.path.join(LOGS_DIR, '{}.json'.format(str(user).split('<')[-1].split('>')[0]))
    print("path:", path)
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


def load_user_dialogue_context(user):
    path = os.path.join(LOGS_DIR, '{}.json'.format(str(user).split('<')[-1].split('>')[0]))
    if not os.path.exists(path):
        return {"choice_answer": "hi, 机器人小金很高兴为您服务", "slot_values": None}
    else:
        with open(path, 'r', encoding='utf8') as f:
            data = f.read()
            return json.loads(data)