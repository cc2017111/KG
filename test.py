import os
import asyncio
from util import gossip_robot, medical_robot, classifier
from utils.json_uitls import dump_user_dialogue_context, load_user_dialogue_context

from wechaty import (
    Contact,
    FileBox,
    Message,
    Wechaty,
    ScanStatus,
)
os.environ["WECHATY_PUPPET"] = "wechaty-puppet-service"
os.environ["WECHATY_PUPPET_SERVICE_TOKEN"] = "12345"
os.environ["WECHATY_PUPPET_SERVICE_ENDPOINT"] = "192.168.31.188:9099"

async def on_message(msg: Message):
    """
    Message Handler for the Bot
    """
    user_intent = classifier(msg.text())
    print(user_intent)
    if msg.is_self():
        return
    if user_intent in ["greet","goodbye","deny","isbot"]:
        reply = gossip_robot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(msg.talker())
        reply = reply.get("choice_answer")
    else:
        reply = medical_robot(msg.text(), msg.talker())
        print(msg.talker())
        print("reply:", reply)
        if reply["slot_values"]:
            dump_user_dialogue_context(msg.talker(), reply)
        reply = reply.get("replay_answer")
    await msg.say(reply)
# async def on_message(msg: Message):
#     """
#     Message Handler for the Bot
#     """
#     if msg.text() == 'ding':
#         await msg.say('dong')
#
#         file_box = FileBox.from_url(
#             'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/'
#             'u=1116676390,2305043183&fm=26&gp=0.jpg',
#             name='ding-dong.jpg'
#         )
#         await msg.say(file_box)


async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    """
    Scan Handler for the Bot
    """
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    """
    Login Handler for the Bot
    """
    print(user)
    # TODO: To be written


async def on_logout(user: Contact):
    """
    Login Handler for the Bot
    """
    print(user)
    # TODO: To be written



async def main():
    """
    Async Main Entry
    """
    #
    # Make sure we have set WECHATY_PUPPET_SERVICE_TOKEN in the environment variables.
    #
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()
    # bot.on('scan',      on_logout)
    bot.on('scan',      on_scan)
    bot.on('login',     on_login)
    bot.on('message',   on_message)

    await bot.start()

    print('[Python Wechaty] Ding Dong Bot started.')

asyncio.run(main())