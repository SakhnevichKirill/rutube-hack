from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer
import torch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import openai
import pandas as pd
import os
import re
import cv2


class AudioProcesser():
    def __init__(self):
        self.whisper_model = WhisperModel('Systran/faster-whisper-large-v3', device='cuda')
        print('Whisper downloaded')
        self.embedder = HuggingFaceEmbeddings(model_name='sergeyzh/rubert-tiny-turbo')
        self.splitter = SemanticChunker(self.embedder, breakpoint_threshold_type = 'percentile', breakpoint_threshold_amount=87)
        self.embedder_for_tags = HuggingFaceEmbeddings(model_name='deepvk/USER-bge-m3')
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.vsegpt.ru/v1",
        )
        print('Openai connected')
    def generate_tags(self, chunk_description, video_description):

        system_prompt = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, обобщенные теги, сильно не конкретизируй, не додумывай, желательно 1-3 тега. В случае если тебе не нравится текст по любым причинам, ответь пустой строкой. Теги должны быть разделены запятой. Вот название и описание видео, чтобы ты знал контекст: {video_description}"
        prompt = chunk_description
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                n=1,
                stop=None,
            )
            tags = response.choices[0].message.content.strip()
            if not tags:
                return []
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        except openai.error.PermissionDeniedError:
            print(f"Модерация отклонила запрос для чанка: {video_description}")
            return []
    def process_video(self, file, video_description):
        file_tags = set()
        
        # Транскрипция видео
        segments, info = self.whisper_model.transcribe(file, beam_size=5)
        text = ' '.join(segment.text for segment in segments)
        print('Video transcribed')
        # Разделение текста на чанки
        chunked_text = self.splitter.split_text(text)
        print(len(chunked_text))
        
        # Получаем эмбеддинги для чанков
        chunk_embeddings = self.embedder.embed_documents(chunked_text)
        video_description_embedding = self.embedder.embed_query(video_description)
        
        # Считаем сходства между эмбеддингами и описанием видео
        similarities = cosine_similarity(chunk_embeddings, [video_description_embedding])
        similarity_threshold = 0.8
        
        for i, sim in enumerate(similarities):
            if sim > similarity_threshold:
                chunk = chunked_text[i]
                
                tags = self.generate_tags(chunk, video_description)
                
                # Если теги есть, добавляем их в множество
                if tags:
                    for t in tags:
                        file_tags.add(t)
                
        
        return file_tags
    
    def get_nearest_tags(self, file_tags, rutube_tags, video_description, similarity_threshold=0.485, top_n=3):
        if len(file_tags) > 0:
            combined_tags = ', '.join(file_tags)
        else:
            combined_tags = ""
        print(combined_tags)
        combined_tags += f', {video_description}'

        generated_embedding = self.embedder_for_tags.embed_query(combined_tags)
        available_tag_embeddings = self.embedder_for_tags.embed_documents(rutube_tags)
        
        # Вычисляем сходства
        similarities = cosine_similarity([generated_embedding], available_tag_embeddings)[0]
        
        # Собираем теги и их сходства, превышающие порог
        nearest_tags = [(rutube_tags[i], similarities[i]) for i in range(len(rutube_tags)) if similarities[i] > similarity_threshold]
        
        # Сортируем по сходству
        nearest_tags = sorted(nearest_tags, key=lambda x: x[1], reverse=True)

        # Проверяем, есть ли хотя бы один тег, который превышает порог
        if not nearest_tags:
            # Если ни один тег не превышает порог, возвращаем тег с максимальным сходством
            max_similarity_index = similarities.argmax()  # Индекс тега с максимальным сходством
            return [rutube_tags[max_similarity_index]]
        
        # Если есть теги, возвращаем top_n подходящих
        nearest_tags = nearest_tags[:top_n]
        return [tag for tag, similarity in nearest_tags]
