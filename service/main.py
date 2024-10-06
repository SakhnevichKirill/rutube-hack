'''тут пишу правильный сервис, потом удалю другой'''
from fastapi import FastAPI, UploadFile, File, HTTPException
from schemes import TranscribationRequest
from models.audio_pipe import TranscriptionChunker
from models.video_pipe import Caption
from models.llm import LLM
from tempfile import NamedTemporaryFile, TemporaryDirectory
import httpx
import json
import os
import time
import tempfile
import moviepy.editor as mp
import uvicorn

import os
import json
import random
from typing import Text
import numpy as np
import rx.operators as ops
from websocket import WebSocket
from diart import sources
from diart.utils import encode_audio
import dotenv
import threading
from threading import Event
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
from settings import *

app = FastAPI()
url = 'сервис макса для транскрибации'

target_tags = [] # потом будем получать из бд
tag_path = '../tag_list.txt'
with open(tag_path, 'r', encoding='utf-8') as file:
    for line in file:
        target_tags.append(line.strip())
chunker = TranscriptionChunker()
video_processer = Caption()
api_key = os.environ.get('OPENAI_API_KEY')
llm = LLM(api_key=api_key)

# Загрузка пути к папке с аудио
AUDIO_FOLDER_PATH = os.environ.get("AUDIO_FOLDER_PATH")
AUDIO_FOLDER_PATH = AUDIO_FOLDER_PATH.replace("\\", "/")
if AUDIO_FOLDER_PATH[-1] == "/":
    AUDIO_FOLDER_PATH = AUDIO_FOLDER_PATH[:-1]


class AssistantManager:
    def __init__(self):
        self.buffer = ""
        self.send_progress = 0
        self.recieve_progress = 0
        self.is_completed_progress = False
        self.receive_complete_event = Event()
        self.send_complete_event = Event()

    def listen_server(self, ws, should_continue):
        """
        Прослушивание сообщений от сервера через WebSocket.
        """
        try:
            while should_continue.is_set():
                message = ws.recv()  # Получаем сообщение
                output = json.loads(message)  # Пробуем распарсить JSON
                # Проверка на наличие ответа и аудио
                if "result" in output:
                    transcribations_json = output.get('result', [])
                    transcribations = [json.loads(x) for x in transcribations_json]
                    self.buffer = transcribations
                    self.update_receive_progress()
                if 'annotations' in output:
                    # Получили неинформативную строку с аннотациями
                    continue
            
        except Exception as e:
            print(f"Error while receiving message: {e}")


    def update_send_progress(self, *args):
        self.send_progress += 1
        print(f"Send progress updated: {self.send_progress}")
        return args

    def update_receive_progress(self, *args):
        self.recieve_progress += 1
        print(f"Receive progress updated: {self.recieve_progress}")
        # Проверяем оба условия: завершение отправки и получение всех данных
        if self.recieve_progress == self.send_progress and self.send_complete_event.is_set():
            self.is_completed_progress = True
            self.receive_complete_event.set()

    def wait_for_completion(self):
        self.receive_complete_event.wait()  # Ждем завершения процесса
    
    def reset_progress(self):
        self.recieve_progress = 0
        self.send_progress = 0
        self.is_completed_progress = False

    def send_audio(self, ws: WebSocket, source: Text, step: float, sample_rate: int):
        """
        Отправка аудио данных на сервер через WebSocket.
        """
        try:
            if not source:
                print("Error: No audio source provided.")
                return

            # Create audio source
            source_components = source.split(":")
        
            # Проверка, если source не "microphone"
            if source_components[0] != "microphone":
                audio_source = sources.FileAudioSource(
                    source, sample_rate, block_duration=step)
            else:
                device = int(source_components[1]) if len(
                    source_components) > 1 else None
                audio_source = sources.MicrophoneAudioSource(step, device)

            # Encode audio, then send through websocket
            audio_source.stream.pipe(
                ops.map(sources.utils.encode_audio),
                ops.do_action(on_next=self.update_send_progress)
            ).subscribe_(
                ws.send, 
                on_completed=self.mark_send_complete  # Вызов при завершении отправки
            )
            audio_source.read()  # Начинаем чтение аудио
            print(f"Audio data sent successfully.")
        except Exception as e:
            print(f"Error while sending message: {e}")

    def mark_send_complete(self):
        """Вызывается, когда отправка всех данных завершена."""
        print("Send process completed!")
        self.send_complete_event.set()  # Сигнализируем о завершении отправки


class AssistantClient:
    def __init__(self, HOST="0.0.0.0", PORT=5002):
        # Инициализация менеджера ассистента
        self.manager = AssistantManager()
        # Флаг для управления потоком
        self.should_continue = threading.Event()
        self.should_continue.set()
        self.HOST = HOST
        self.PORT = PORT

    def start(self, audio_source="microphone"):
        """
        Запуск клиента для прослушивания и отправки аудио через WebSocket.
        """
        try:
            # Подключение к серверу через WebSocket
            print("Connecting to server...")
            self.ws = WebSocket()
            self.ws.connect(f"ws://{self.HOST}:{self.PORT}")

            # Запуск потока для прослушивания сообщений от сервера
            receiver = threading.Thread(
                target=self.manager.listen_server, args=(self.ws, self.should_continue))
            receiver.start()

            # Создание источника аудио и запуск потока для отправки аудио
            # audio_source = "./INIT_WARM.wav"
            # audio_source = "microphone"
            sender = threading.Thread(
                target=self.manager.send_audio, args=(self.ws, audio_source, PIPELINE_STEP, PIPELINE_SAMPLE_RATE))
            sender.start()

        except:
            print("Exiting...")
            self.should_continue.clear()
            self.ws.close()
            receiver.join()
            sender.join()
            audio_source.close()


client = AssistantClient()

@app.post("/get_tags")
async def get_tags(video_description: str, file: UploadFile = File(...)):
    with TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
    audio_path = '../INIT_WARM_splited.wav'
    client.start(audio_path)
    if client:
        print('Client started')
    client.manager.receive_complete_event.wait()
    transcription_data = client.manager.buffer[0]['text']
    print(transcription_data)


    
    tags_for_filter = video_description + ''' Транспорт, Книги и литература, Бизнес и финансы, Карьера, Образование,
                                           События и достопримечательности, Семья и отношения, Изобразительное искусство,
                                           Еда и напитки, Здоровый образ жизни, Хобби и интересы, Дом и сад,
                                           Медицина, Фильмы и анимация, Музыка и аудио, Новости и политика,
                                           Личные финансы, Животные, Массовая культура, Недвижимость,
                                           Религия и духовность, Наука, Покупки, Спорт, Стиль и красота,
                                           Информационные технологии, Телевидение, Путешествия, Игры'''
    #в случае с кейсом рутуба целевые теги для фильтров состоят из тайтл + дескрипшн + целевые теги первого уровня
    # теперь нужно получить чанки, сохранив таймкоды

    filtered_chunks = chunker.process_transcription_request(transcription_data, tags_for_filter)
    # тут хочу сделать, чтобы в video_pipe была функция, которая может сделать описание по таймкоду:
    # start_time-end_time, а не по всему видосу
    # video_caption = video_processer.shot_transit('../test_videos/trimmed_1.mp4')
    # data = json.loads(video_caption)
    # video_caption =" ".join(item['caption'] for item in data)
    tags = set()
    print('Чанки: ', len(filtered_chunks))
    for audio_chunk in filtered_chunks:
        tags_for_chunk = llm.generate_tags(audio_chunk, tags_for_filter)
        tags.update(tags_for_chunk)
    print(tags)
    # получили множество тегов к каждому чанку, теперь переводим их в целевые (target_tags)
    final_tags = chunker.get_nearest_tags(tags, target_tags, similarity_threshold=0.7, top_n=2)
    return final_tags

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9000)