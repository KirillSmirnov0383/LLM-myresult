from flask import Flask, render_template, redirect, url_for
import json
import os
from datetime import datetime

class ChatApp:
    def __init__(self, app: Flask, message_folder: str):
        self.app = app
        self.message_folder = message_folder
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            # Получаем отсортированный список файлов
            files = self.get_files_sorted_by_date(self.message_folder)
            return render_template('index.html', files=files)

        @self.app.route('/chat/<filename>')
        def chat(filename):
            # Загружаем переписку из выбранного файла JSON
            with open(os.path.join(self.message_folder, filename), 'r', encoding='utf-8') as file:
                messages = json.load(file)
            return render_template('chat.html', messages=messages)

    def get_files_sorted_by_date(self, folder):
        # Получаем список файлов в папке и сортируем их по времени изменения
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
        return files

if __name__ == '__main__':
    app = Flask(__name__)
    message_folder = 'chat_history'
    chat_app = ChatApp(app, message_folder)
    app.run(debug=True)