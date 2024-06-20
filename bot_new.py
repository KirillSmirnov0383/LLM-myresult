import asyncio
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, Application, ConversationHandler, ContextTypes

from main import LAI
from my_calendar import CalendarClient
import os

import datetime

import time

TOKEN = os.getenv("TOKEN")

STATE_DATE = 1

class TelegramBot:
    def __init__(self):
        self.updater = Application.builder().token(TOKEN).build()
        self.cal = CalendarClient()
        self.lai = LAI()
        self.start_time = time.time()

    async def start(self, update, context):
        keyboard = [["Записаться"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Я бот. Как дела?", reply_markup=reply_markup)

    async def reload(self, update, context):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="LAI успешно перезагружен.")

    async def subscribe(self, update, context):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="На какую дату вы хотите записаться? Введите сообщение в формате (Год, месяц, день, часы, минуты, секунды) Пример (2024, 9, 12, 15, 0, 0)")
        return STATE_DATE

    async def handle_subscription_date(self, update, context):
        selected_date = update.message.text
        name = update.effective_user.name
        y, m, d, h, min, sec =map(int, selected_date.split(', '))
        start_time = datetime.datetime(y, m, d, h, min, sec, tzinfo=datetime.timezone.utc)
        await update.message.reply_text(self.cal.create_google_calendar_event(update.effective_user.name, start_time))
        return -1
    
    async def textMessage(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message.date.timestamp() < self.start_time:
            return
        response = ''
        while response == '':
            response, chunk = self.lai.answer_index(update.message.text)
            print("отправил запрос")
        chunk = "------------------".join(chunk)
        await update.message.reply_text(response)
        context.chat_data[update.message.text] = response
        print(context.chat_data,"\n")

    def run(self):
        self.updater.add_handler(CommandHandler('start', self.start))
        self.updater.add_handler(CommandHandler('reload', self.reload))
        self.updater.add_handler(CommandHandler('subscribe', self.subscribe))
        subscribe_handler = ConversationHandler(
            entry_points=[MessageHandler(filters.Regex('^Записаться$'), self.subscribe)],
            states={
                STATE_DATE: [MessageHandler(filters.TEXT, self.handle_subscription_date)]
            },
            fallbacks=[]
        )

        self.updater.add_handler(subscribe_handler)
        self.updater.add_handler(MessageHandler(filters= None, callback= self.textMessage))

        #self.updater.add_handler(MessageHandler(filters.Regex(r'^Записаться$'), self.subscribe))
        #self.updater.add_handler(MessageHandler(filters.Regex(r'^Записаться$'), self.handle_subscription_date))
        

        self.updater.run_polling()


if __name__ == '__main__':
    bot = TelegramBot()
    bot.run()
