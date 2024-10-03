from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer
import torch
import csv
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
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)
import numpy as np
import json
from moviepy.editor import *

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
        api_key = 'sk-or-vv-ed9b0360080c7f2ebfea7bb6d596cc6f67d2d2d8356b6cd950e66379b63c2d87'
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


class VideoCaptioningModel:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", 
                                                          torch_dtype=self.torch_dtype, 
                                                          trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", 
                                                       trust_remote_code=True)

    def get_caption(self, 
                    image: np.array,
                    task_prompt: str='<MORE_DETAILED_CAPTION>'):
        height, width, _ = image.shape
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(width, height)
        )
        return parsed_answer[task_prompt]

class Caption():
    def __init__(self):
        self.video_captioning_model = VideoCaptioningModel()
    
    def calculate_histogram(self, frame):
        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Calculate histogram
        hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
        # Normalize the histogram
        cv2.normalize(hist, hist)
        return hist.flatten()

    def shot_transit(self, 
                     input_file,
                     threshold: int=0.7):
        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

        previous_hist = None
        scene_changes = [0.0]
        frame_list = []
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_hist = self.calculate_histogram(frame)
            if previous_hist is not None:
                # Use correlation to compare histograms
                similarity = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)

                # Scene change threshold
                if similarity < threshold:
                    scene_changes.append(round((int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1) / fps, 2))  # Log time of new scene start
                    frame_list.append(frame) # Add first scene frame to list
            else:
                frame_list.append(frame)

            # Update previous histogram
            previous_hist = current_hist
        
        # get captions of each scene
        for frame, start_time in zip(frame_list, scene_changes):
            caption = self.video_captioning_model.get_caption(frame)
            results.append({'scene_start': start_time, 'caption': caption})
        return json.dumps(results, ensure_ascii=False)

    def shot_transit_uniform(self, input_file):
        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        time_intervals = duration / 12

        results = []
        timecode = 0.0

        while timecode <= duration:
            end_time = timecode + time_intervals
            if end_time > duration:
                end_time = duration

            cap.set(cv2.CAP_PROP_POS_FRAMES, (timecode + (time_intervals / 2)) * fps)
            ret, frame = cap.read()

            if not ret:
                break

            caption = self.video_captioning_model.get_caption(frame)
            results.append({'scene_start': timecode, 'caption': caption})
            timecode += time_intervals
        return json.dumps(results, ensure_ascii=False)

def find_tags(df, target_tags):
    collected_tags = {}
    
    for tag in target_tags:
        if tag in df['Уровень 3 (iab)'].values:
            rows = df[df['Уровень 3 (iab)'] == tag]
            if not rows.empty:
                for _, row_data in rows.iterrows():
                    level1 = row_data['Уровень 1 (iab)']
                    level2 = row_data['Уровень 2 (iab)']
                    level3 = row_data['Уровень 3 (iab)']
                    if level1 not in collected_tags:
                        collected_tags[level1] = {}
                    if level2 not in collected_tags[level1]:
                        collected_tags[level1][level2] = set()
                    collected_tags[level1][level2].add(level3)
                    
        elif tag in df['Уровень 2 (iab)'].values:
            rows = df[df['Уровень 2 (iab)'] == tag]
            if not rows.empty:
                for _, row_data in rows.iterrows():
                    level1 = row_data['Уровень 1 (iab)']
                    level2 = row_data['Уровень 2 (iab)']
                    if level1 not in collected_tags:
                        collected_tags[level1] = {}
                    if level2 not in collected_tags[level1]:
                        collected_tags[level1][level2] = set()
                    
        elif tag in df['Уровень 1 (iab)'].values:
            level1 = tag  # Здесь tag - это уровень 1
            if level1 not in collected_tags:
                collected_tags[level1] = {}

    # Преобразуем собранные теги в нужный формат
    result = []
    for level1, level2_tags in collected_tags.items():
        if not level2_tags:
            # Если нет связанных тегов 2 уровня, просто добавляем тег 1 уровня
            result.append(level1)
        else:
            for level2, level3_tags in level2_tags.items():
                result.append(f'{level1}: {level2}')  # Добавляем уровень 2
                if level3_tags:
                    for level3 in level3_tags:
                        result.append(f'{level1}: {level2}: {level3}')  # Добавляем уровень 3

    return ', '.join(result)

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def trim_video(input_file, output_file, start_time, end_time):
    try:
        # Обрезаем видео с использованием moviepy
        ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)
        print(f"Сохранено обрезанное видео: {output_file}")
    except Exception as e:
        print(f"Произошла ошибка при обрезке видео: {str(e)}")


if __name__ == '__main__':
    audio_processer = AudioProcesser()
    video_processer = Caption()
    df = pd.read_csv('/home/ubuntu/rutube-hack/test_tag_video (2)/IAB_tags.csv')
    video_dir = '/home/ubuntu/rutube-hack/test_tag_video (2)/videos'
    output_file = 'output_tags.csv'  # Путь к выходному CSV файлу

    # Открываем CSV файл для записи результатов
    with open(output_file, mode='w', encoding='utf-8', newline='') as output_csv:
        csv_writer = csv.writer(output_csv)
        # Записываем заголовок
        csv_writer.writerow(['video_id', 'result_tags'])

        # Открываем исходный CSV файл
        with open('/home/ubuntu/rutube-hack/test_tag_video (2)/sample_submission.csv', mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)  # Читаем файл как словарь с заголовками

            # Проходим по каждой строке в CSV
            for row in csv_reader:
                video_id = row['video_id']
                title = row['title']
                description = row['description']
                video_description = f'{title}. {description}'
                video_file_path = os.path.join(video_dir, f'{video_id}.mp4')
                output_video_file_path = os.path.join(video_dir, f'trimmed_{video_id}.mp4')
                trim_video(video_file_path, output_video_file_path, 0, 130)
                video_file_path = output_video_file_path
                if os.path.exists(video_file_path):
                    tags = audio_processer.process_video(video_file_path, video_description)
                    video_caption = video_processer.shot_transit(video_file_path)
                    data = json.loads(video_caption)
                    video_caption =" ".join(item['caption'] for item in data)
                    video_tags = audio_processer.generate_tags(video_caption, video_description)
                    tags.update(video_tags)
                    print(tags)
                    rutube_tags = []
                    tag_path = 'tag_list.txt'
                    
                    with open(tag_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            rutube_tags.append(line.strip())
                    
                    # Получаем результат тегов
                    result_tags = audio_processer.get_nearest_tags(tags, rutube_tags, video_description)
                    print(result_tags)
                    res = find_tags(df, result_tags)
                    print(res)
                    # Записываем video_id и результат тегов в новый CSV
                    csv_writer.writerow([video_id, res])  # Объединяем теги в строку
            