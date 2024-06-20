from openai import OpenAI
import json
import requests

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import re
import tiktoken



class LAI():
    def __init__(self):
        self.url = "http://serv.sae.ru:8888/v1"
        self.api = "not-needed"
        self.data = "https://docs.google.com/document/d/1gCvcpAgRrVjON801fBgwPBYyo46b3xT41bvxR_hN_b4/edit"
        self.read_source_documents(self.data)
        
    def create_embedding(self, text_data: str) -> None:
        """
        Функция create_embedding принимает текстовые данные в виде строки и создает эмбеддинги для каждого куска текста.
        Она разбивает входные данные на куски, создает для каждого куска объект Document,
        использует модель эмбеддингов для преобразования текста в векторы, а затем сохраняет эти векторы в объект Chroma и сохраняет объект в файл 'chroma.pkl' для дальнейшего использования..
        Функция также сохраняет объект Chroma в файл 'chroma.pkl' для дальнейшего использования.
        """
        source_documents = []
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)

        for chunk in text_splitter.split_text(text_data):
            source_documents.append(Document(page_content=chunk, metadata={}))

        embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.chroma_db = Chroma.from_documents(source_documents, embedding_model)

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

    def sendRequest(self, system, promt):
        client = OpenAI(base_url=self.url, api_key=self.api)
        response = client.chat.completions.create(
            model="StableBeluga-13B-GGUF/stablebeluga-13b.Q8_0.gguf",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": promt}
            ],
            temperature=0.5,
            max_tokens=1000,
            timeout=30
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
    
    def answer_index(self, system, topic, temp = 1, verbose = 0):       
        chunk = []
        docs = self.chroma_db.similarity_search(topic, k=1)
        for i, doc in enumerate(docs):
            chunk.append(doc.page_content)

        message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
        print(self.getAnswer(self.sendRequest(system + f"{message_content}", topic)))

    


    
lai = LAI()

marketing_index = lai.load_search_indexes('https://docs.google.com/document/d/1gCvcpAgRrVjON801fBgwPBYyo46b3xT41bvxR_hN_b4/edit')

text = "Здравствуйте, меня зовут Константин. Какой адрес у вашего магазина. Я живу в Южном регионе"


marketing_chat_promt = '''Ты менеджер поддержки в чате компании, компания продает товары разного назначения. 
У тебя есть большой документ со всеми материалами о продуктах компании. 
Тебе задает вопрос клиент в чате, дай ему ответ, опираясь на документ, постарайся ответить так, чтобы человек захотел после ответа купить товар. 
и отвечай максимально точно по документу, не придумывай ничего от себя. 
Документ с информацией для ответа клиенту: '''



lai.answer_index(
    marketing_chat_promt,
    text
)

#print(lai.answer(text))

"""from langchain.embeddings import SelfHostedHuggingFaceEmbeddings
import runhouse as rh
model_name = "sentence-transformers/all-mpnet-base-v2"
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
hf = SelfHostedHuggingFaceEmbeddings(model_name=model_name, hardware=gpu)"""



# Выбор модели и токенизатора
"""model_name = "sentence-transformers/all-mpnet-base-v2"
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