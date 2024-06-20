from openai import OpenAI
import json
import requests
 
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os

import openai
 
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import getpass
import re
import tiktoken

from transformers import AutoModel, AutoTokenizer
 
 
 
class LAI():
    def __init__(self):
        self.url = "http://serv.sae.ru:8888/v1"
        self.api = "not-needed"
        self.data = "https://docs.google.com/document/d/1Dco7E7xhJ1eD2cDLBZjvyZBYJBYNmxGu5gUIxMDzwX4/edit?usp=sharing"
        self.read_source_documents(self.data)
 
    def create_embedding(self, text_data: str) -> None:
        """
        Функция create_embedding принимает текстовые данные в виде строки и создает эмбеддинги для каждого куска текста.
        Она разбивает входные данные на куски, создает для каждого куска объект Document,
        использует модель эмбеддингов для преобразования текста в векторы, а затем сохраняет эти векторы в объект Chroma и сохраняет объект в файл 'chroma.pkl' для дальнейшего использования..
        Функция также сохраняет объект Chroma в файл 'chroma.pkl' для дальнейшего использования.
        """
        #os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
        source_documents = []
        text_splitter = CharacterTextSplitter(separator="#", chunk_size=256, chunk_overlap=0)
    
        for chunk in text_splitter.split_text(text_data):
            source_documents.append(Document(page_content=chunk, metadata={}))
        
 
        embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.chroma_db = Chroma.from_documents(source_documents, embedding_model)
 
    @ZALUPA
    def read_source_documents(self, google_docs_url: str) -> None:
        """
        Читает текстовые документы из Google Docs по заданному URL.
        """
 
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', google_docs_url)
        if match_ is None:
            raise ValueError('Недопустимый URL Google Docs')
        doc_id = match_.group(1)
 
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        text = response.text
        self.create_embedding(text)
 
    def createPromt(self, promt):
        return promt
    
    def sendRequest1(self, system, prompt):

        openai.api_key = "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"  # Замените YOUR_API_KEY на ваш ключ API
    
        response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # Выберите подходящий для ваших целей движок
        prompt=system + prompt,      # Объединяем системный и пользовательский ввод
        temperature=0.5,
        max_tokens=4000,
        timeout=100
        )
        return response

 
    def sendRequest(self, system, promt):
        client = OpenAI(base_url=self.url, api_key="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")
        response = client.chat.completions.create(
            model="saiga_mistral_7b-AWQ",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": promt}
            ],
            temperature=0.5,
            max_tokens=4000,
            timeout=100
        )
        return response
    
    
 
    def getAnswer(self, response):
        response_dict = json.loads(response.json())
        answer = response_dict['choices'][0]['message']['content']
        print(type(response_dict))
        return answer
 
    def answer(self, user_request):
        promt = self.createPromt(user_request)
        answer_json = self.sendRequest(promt)
        return self.getAnswer(answer_json)
 
    def load_search_indexes(self, url: str) -> str:
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
        if match_ is None:
            raise ValueError('Invalid Google Docs URL')
        doc_id = match_.group(1)
 
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        text = response.text
        return self.create_embedding(text)
 
    def answer_index(self, topic, temp = 1, verbose = 0):  
        system = '''Ты консультант по баням. Ты знаешь все о банях, ты отвечаешь на все вопросы связанные с банями. Ты отвечаешь на вопросы строго по документу, ничего не предумываешь от себя.
            Ты консультируешь клиентов компании, которая оказывает только банные услуги. Тебе задает вопрос клиент в чате, дай ему ответ, опираясь на документ, постарайся ответить так, человек получил правильный ответ.
            Отвечай только на русском языке. Клиент не должен ничего знать о документе. Не говори ничего про документ.
            Отвечай максимально точно по документу, не придумывай ничего от себя. Если в документе нет ответа, скажи что не знаешь ответ на этот вопрос. Не добавляй информацию, которой нет в документе.
            Документ с информацией для ответа клиенту: '''     
        chunk = []
        docs = self.chroma_db.similarity_search(topic, k=10)
        print(docs, type(docs))
        for i, doc in enumerate(docs):
            if doc.page_content not in chunk and len(chunk) < 5:
                chunk.append(doc.page_content)
        message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + chunk[i] + '\n' for i in range(len(chunk))]))
        return self.getAnswer(self.sendRequest(system + f"{message_content}", topic)), chunk
 
 
 
 


 
 
""" lai.answer_index(
    marketing_chat_promt,
    text
)""" 
""" 
#print(lai.answer(text))
 
from langchain.embeddings import SelfHostedHuggingFaceEmbeddings
import runhouse as rh
model_name = "text-embedding-ada-002"
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
hf = SelfHostedHuggingFaceEmbeddings(model_name=model_name, hardware=gpu)
 """


""" 
# Выбор модели и токенизатора
model_name = "Xenova/text-embedding-ada-002"

# Загрузка модели
model = AutoModel.from_pretrained(model_name)

model_name = "Xenova/text-embedding-ada-002"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
# Подготовка текста
text = "Ваш текст здесь."
 
# Токенизация текста
tokens = tokenizer(text, return_tensors="pt")
 
# Получение векторного представления
outputs = model(**tokens)
vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
 
print(vector)"""