'''тут пишу правильный сервис, потом удалю другой'''
from fastapi import FastAPI, UploadFile, File, HTTPException
from .schemes import TranscribationRequest
from .models.audio_pipe import TranscriptionChunker
from .models.video_pipe import Caption
from .models.llm import LLM
from tempfile import NamedTemporaryFile, TemporaryDirectory
import httpx
import json
import os

import os
import json
import random
from typing import Text
import numpy as np
import rx.operators as ops
from websocket import WebSocket, create_connection
from diart import sources
from diart.utils import encode_audio
import dotenv
import threading
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
from settings import *

app = FastAPI()
url = 'сервис макса для транскрибации'

target_tags = [] # потом будем получать из бд
tag_path = 'tag_list.txt'
with open(tag_path, 'r', encoding='utf-8') as file:
    for line in file:
        target_tags.append(line.strip())
chunker = TranscriptionChunker()
video_processer = Caption()
llm = LLM()

# Загрузка пути к папке с аудио
AUDIO_FOLDER_PATH = os.environ.get("AUDIO_FOLDER_PATH")
AUDIO_FOLDER_PATH = AUDIO_FOLDER_PATH.replace("\\", "/")
if AUDIO_FOLDER_PATH[-1] == "/":
    AUDIO_FOLDER_PATH = AUDIO_FOLDER_PATH[:-1]


class AssistantManager:
    def __init__(self):
        # Определение доступных действий
        self.talk_moves = {
            'negative_valence_monitoring_cognition': [
                "feel_stuck", "better_understanding", "reflecting", "current_approach"
            ],
            'issue_conceptual_understanding': [
                "confusion", "key_concepts", "tools"
            ],
            'lack_shared_perspective': [
                "progress_group", "joint_plan", "accomplish_group"
            ],
        }

        # Организация путей к аудио файлам
        self.audio_files = {}
        for talk_move in self.talk_moves:
            self.audio_files[talk_move] = {}
            for variation in self.talk_moves[talk_move]:
                self.audio_files[talk_move][variation] = f"{AUDIO_FOLDER_PATH}/{variation}.mp3"

        # Инициализация Text-to-Speech (tts) движка
        self.tts = pyttsx3.init()

        # Установка свойств перед добавлением текста для озвучивания
        # Скорость озвучивания (может превышать 100)
        self.tts.setProperty('rate', 130)
        self.tts.setProperty('volume', 0.9)  # Громкость от 0 до 1
        voices = self.tts.getProperty('voices')
        self.tts.setProperty('voice', voices[1].id)  # Установка голоса

        self.buffer = ""
        self.send_progress = 0
        self.recieve_progress = 0
        self.is_completed_progress = False


    def play_audio(self, talk_move):
        """
        Проигрывание аудио файла, связанного с конкретным действием.
        """
        try:
            # Выбор случайной вариации действия
            variation = random.choice(self.talk_moves[talk_move])
            file_path = self.audio_files[talk_move][variation]
            # Проигрывание аудио
            play(AudioSegment.from_mp3(file_path))
        except:
            print(f"ERROR: Audio file not found for the talk_move {talk_move}")
            pass

    def say(self, text):
        """
        Озвучивание текста с помощью TTS.
        """
        self.tts.say(text)

    def runAndWait(self):
        """
        Запуск TTS и ожидание завершения озвучивания.
        """
        self.tts.runAndWait()

    def listen_server(self, ws, should_continue):
        """
        Прослушивание сообщений от сервера через WebSocket.
        """
        try:
            while should_continue.is_set():
                message = ws.recv()
                #print(f"Received: {message}", end="")
                self.buffer = message
                self.update_recieve_progress()
                
        except Exception as e:
            print(f"Error while receiving message: {e}")
        return 
        try:
            while should_continue.is_set():
                # Получение сообщения от сервера
                output = ws.recv()
                # Преобразование сообщения в словарь
                output = json.loads(output)
                print(output)

                # Проверка наличия ответа
                if output['response']:
                    # Воспроизведение ответа в виде аудио
                    try:
                        self.play_audio(output['selected_move'])
                    except:
                        print(
                            f"WARNING: Audio file not found: {output['response']}")
                        pass  # Пока не воспроизводим никакое аудио
                elif 'test test' in output['transcription'].lower().replace(",", ""):
                    self.play_audio('issue_conceptual_understanding')

        except Exception as e:
            print(f"Error while receiving message: {e}")

    def update_send_progress(self, *args):
        self.send_progress += 1
        return args

    def update_recieve_progress(self, *args):
        self.recieve_progress += 1
        if self.recieve_progress == self.send_progress:
            self.is_completed_progress = 1
        return args
    
    def reset_progress(self):
        self.recieve_progress = 0
        self.send_progress = 0
        self.is_completed_progress = False

    def send_audio(self, ws: WebSocket, source: Text, step: float, sample_rate: int):
        """
        Отправка аудио данных на сервер через WebSocket.
        """
        try:
            # Create audio source
            source_components = source.split(":")
            if source_components[0] != "microphone":
                audio_source = sources.FileAudioSource(
                    source, sample_rate, block_duration=step)
            else:
                device = int(source_components[1]) if len(
                    source_components) > 1 else None
                audio_source = sources.MicrophoneAudioSource(step, device)

            # Encode audio, then send through websocket
            audio_source.stream.pipe(
                ops.map(self.update_send_progress),
                ops.map(sources.utils.encode_audio)).subscribe_(ws.send)

            # Start reading audio
            print(f"Listening to audio_source: {type(audio_source)}...")
            audio_source.read()
        except Exception as e:
            print(f"Error while sending message: {e}")

class AssistantClient:
    def __init__(self, HOST="127.0.0.1", PORT=5002):
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

        except KeyboardInterrupt:
            print("Exiting...")
            self.should_continue.clear()
            self.ws.close()
            receiver.join()
            sender.join()
            audio_source.close()

client = AssistantClient()


@app.post("/get_tags")
async def get_tags(video_description: str, file: UploadFile = File(...)):
    '''Сохраняем видос в временную папку, чтобы потом его можно было обработать.
    Потом тут будет сохранение в s3, когда доделаю.'''
    with TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
    try:

        client.start(temp_file_path)
        
        while not client.manager.is_completed_progress:
            transcription_data = client.manager.buffer
            print(f"{client.manager.recieve_progress / client.manager.send_progress} proccessing...")
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(url, file=temp_file_path)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error from transcription service: {response.text}")
        """
        
        # transcription_data = response.json()

        '''Валидируем на соответсвие схеме'''
        try:
            validated_data = TranscribationRequest(**transcription_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid transcription data format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    tags_for_filter = video_description + [''' Транспорт, Книги и литература, Бизнес и финансы, Карьера, Образование,
                                           События и достопримечательности, Семья и отношения, Изобразительное искусство,
                                           Еда и напитки, Здоровый образ жизни, Хобби и интересы, Дом и сад,
                                           Медицина, Фильмы и анимация, Музыка и аудио, Новости и политика,
                                           Личные финансы, Животные, Массовая культура, Недвижимость,
                                           Религия и духовность, Наука, Покупки, Спорт, Стиль и красота,
                                           Информационные технологии, Телевидение, Путешествия, Игры''']
    #в случае с кейсом рутуба целевые теги для фильтров состоят из тайтл + дескрипшн + целевые теги первого уровня
    # теперь нужно получить чанки, сохранив таймкоды
    filtered_chunks = chunker.process_transcription_request(validated_data, tags_for_filter)
    # тут хочу сделать, чтобы в video_pipe была функция, которая может сделать описание по таймкоду:
    # start_time-end_time, а не по всему видосу
    video_caption = video_processer.shot_transit(temp_file_path)
    data = json.loads(video_caption)
    video_caption =" ".join(item['caption'] for item in data)
    tags = set()
    for audio_chunk in filtered_chunks:
        tags_for_chunk = llm.generate_tags(audio_chunk, video_caption, tags_for_filter)
        tags.add(tags_for_chunk)
    # получили множество тегов к каждому чанку, теперь переводим их в целевые (target_tags)
    final_tags = chunker.get_nearest_tags(tags, target_tags, video_description, similarity_threshold=0.7, top_n=2)
    return final_tags