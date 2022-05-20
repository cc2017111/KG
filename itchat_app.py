import os
import itchat
from util import gossip_robot, medical_robot, classifier
from utils.json_uitls import dump_user_dialogue_context, load_user_dialogue_context


def delete_cache(file_name):
    """ 清除缓存数据，切换账号登入 """
    if os.path.exists(file_name):
        os.remove(file_name)


@itchat.msg_register(['Text'])
def text_replay(msg):
    user_intent = classifier(msg['Text'])
    print(user_intent)
    if user_intent in ["greet","goodbye","deny","isbot"]:
        reply = gossip_robot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(msg.User['NickName'])
        reply = reply.get("choice_answer")
    else:
        reply = medical_robot(msg['Text'],msg.User['NickName'])
        if reply["slot_values"]:
            dump_user_dialogue_context(msg.User['NickName'],reply)
        reply = reply.get("replay_answer")

    msg.user.send(reply)


if __name__ == '__main__':
    # delete_cache(file_name='./logs/loginInfo.pkl')
    itchat.auto_login(hotReload=True, enableCmdQR=2, statusStorageDir='./logs/loginInfo.pkl')
    itchat.run()