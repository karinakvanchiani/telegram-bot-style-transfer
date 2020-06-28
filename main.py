from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import datetime

import torchvision.models as models
import torch

from utils import run_style_transfer, image_loader
from config import TG_TOKEN, TG_API_URL


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

cnn = models.vgg19(pretrained=True).features.eval()


def do_start(bot: Bot, update: Update):

    bot.send_message(chat_id=update.message.chat_id, text='Bot is ready! Напишите приветствие')


def send_photo(bot, chat_id):

    style = image_loader('/content/gdrive/My Drive/pictures/style.jpg')
    content = image_loader('/content/gdrive/My Drive/pictures/content.jpg')
    input_img = content.clone()

    photo = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                               content, style, input_img, num_steps=30)

    bot.send_photo(chat_id=chat_id, photo=photo)


def save_photo(text, photo):

    if text == 'ready content':
        photo.save('/content/gdrive/My Drive/pictures/content.jpg')

    else:
        photo.save('/content/gdrive/My Drive/pictures/style.jpg')


def send_message(bot: Bot, update: Update):

    greetings = ('здравствуй', 'привет', 'hey', 'hello')

    now = datetime.datetime.now()
    hour = now.hour

    chat_id = update.message.chat_id
    text = update.message.text

    if text.lower() in greetings:
        if 6 <= hour < 12:
            bot.send_message(chat_id, 'Доброе утро!\nНажмите /style_transfer для стилизации Вашего фото')

        elif 12 <= hour < 17:
            bot.send_message(chat_id, 'Добрый день!\nНажмите /style_transfer для стилизации Вашего фото')

        elif 17 <= hour < 23:
            bot.send_message(chat_id, 'Добрый вечер!\nНажмите /style_transfer для стилизации Вашего фото')

    else:
        if text == '/style_transfer':
            text = 'пришлите фото контента и напишите "ready content" после отправки'

        elif text == 'ready content':
            photo = update.message.photo[-1]
            save_photo(text, photo)
            text = 'пришлите фото стиля и напишите "ready style" после отправки'

        elif text == 'ready style':
            photo = update.message.photo[-1]
            save_photo(text, photo)
            text = 'ожидайте результат'

        elif text == 'ожидайте результат':
            send_photo(bot, chat_id)
            text = 'the end'

        bot.send_message(chat_id=chat_id, text=text)


def main():

    bot = Bot(token=TG_TOKEN, base_url=TG_API_URL)
    updater = Updater(bot=bot)

    start_handler = CommandHandler('start', do_start)
    updater.dispatcher.add_handler(start_handler)

    message_handler = MessageHandler(Filters.text, send_message)
    updater.dispatcher.add_handler(message_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
