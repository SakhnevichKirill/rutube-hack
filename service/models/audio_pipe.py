from typing import List
from schemes import TranscribationItem, TranscribationRequest
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os


class TranscriptionChunker:
    def __init__(self, model_name: str = 'sergeyzh/rubert-tiny-turbo', threshold: float = 0.5):
        '''
        Инициализация Chunker с моделью для генерации эмбеддингов и порогом для косинусной близости.
        
        :param model_name: Название модели для SentenceTransformer.
        :param threshold: Порог косинусной близости для объединения транскрипций.
        '''
        self.model = SentenceTransformer(model_name)
        self.semantic_embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.threshold = threshold
        self.chunker = SemanticChunker(self.semantic_embedder, breakpoint_threshold_type = 'percentile', breakpoint_threshold_amount=85)

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        '''Вычисляем косинусную близость между двумя текстами с помощью эмбеддингов.'''
        # Получаем эмбеддинги для текстов
        embeddings1 = self.model.encode([text1])
        embeddings2 = self.model.encode([text2])
        
        # Вычисляем косинусную близость между эмбеддингами
        similarity = cosine_similarity(embeddings1, embeddings2)
        
        return similarity[0][0]  # Возвращаем значение косинусной близости

    def chunk_transcriptions(self, transcriptions: List[TranscribationItem]) -> List[TranscribationItem]:
        '''Функция для объединения транскрипций на основе косинусной близости.'''
        chunked_transcriptions = self.chunker.split_text(transcriptions)
        return chunked_transcriptions

    def filter_chunks(self, chunks: List[TranscribationItem], target_tags: str) -> List[TranscribationItem]:
        '''Фильтруем чанки на основе целевых тегов'''
        filtered_chunks = []

        for chunk in chunks:
            '''TODO
            добавить векторную бд, сделать фильтрацию опциональной, если нет целевых тегов
            '''
            similarity = self.calculate_cosine_similarity(chunk, target_tags)
            if similarity >= self.threshold:  # Сравниваем с порогом
                filtered_chunks.append(chunk)

        return filtered_chunks

    def process_transcription_request(self, request: TranscribationRequest, target_tags) -> TranscribationRequest:
        '''Функция для обработки запроса и возврата отфильтрованных и chunked транскрипций'''
        chunked_transcriptions = self.chunk_transcriptions(request)
        filtered_chunks = self.filter_chunks(chunked_transcriptions, target_tags)
        return filtered_chunks
  
    
    def get_nearest_tags(
        self,
        file_tags: set,  # Теперь file_tags принимает множество
        target_tags: List[str], 
        similarity_threshold: float = 0.485, 
        top_n: int = 3
    ) -> List[str]:
        # Формируем объединённые теги
        combined_tags = ', '.join(file_tags)  # Преобразуем множество в строку

        # Генерируем эмбеддинг для объединённых тегов
        generated_embedding = self.model.encode([combined_tags])
        # Генерируем эмбеддинги для доступных тегов
        available_tag_embeddings = self.model.encode(target_tags)
        
        # Вычисляем сходства
        similarities = cosine_similarity(generated_embedding, available_tag_embeddings)[0]
        
        # Собираем теги и их сходства, превышающие порог
        nearest_tags = [(target_tags[i], similarities[i]) for i in range(len(target_tags)) if similarities[i] > similarity_threshold]
        
        # Сортируем по сходству
        nearest_tags = sorted(nearest_tags, key=lambda x: x[1], reverse=True)

        # Проверяем, есть ли хотя бы один тег, который превышает порог
        if not nearest_tags:
            # Если ни один тег не превышает порог, возвращаем тег с максимальным сходством
            max_similarity_index = similarities.argmax()  # Индекс тега с максимальным сходством
            return [target_tags[max_similarity_index]]
        
        # Если есть теги, возвращаем top_n подходящих
        nearest_tags = nearest_tags[:top_n]
        return [tag for tag, similarity in nearest_tags]
