from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import pandas as pd
import os

video_path = '/home/ubuntu/rutube-hack/test_videos'
whisper_model = WhisperModel('Systran/faster-whisper-large-v3', device='cuda')
print('WHISPER_UPLOADED')

summarizator = pipeline(task="summarization", model='cointegrated/rut5-base-absum', device = 'cuda')
print('SUMMARIZATOR_UPLOADED')
api_key = 'sk-or-vv-ed9b0360080c7f2ebfea7bb6d596cc6f67d2d2d8356b6cd950e66379b63c2d87'
client = OpenAI(
    api_key=api_key,
    base_url="https://api.vsegpt.ru/v1",
)
def generate_tags(video_description):
    system_prompt = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, не додумывай, желательно 1-3 тега. Теги должны быть разделены запятой. Ты можешь использовать только теги из этого списка: {rutube_tags}. Если нет подходящих то выведи тег nan. От качества тегов зависит моя работа!"
    prompt = video_description
    messages = [{"role":"system", "content":system_prompt}]
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        n=1,            # Number of responses
        stop=None,      # Stopping condition
    )
    tags = response.choices[0].message.content
    return [tag.strip() for tag in tags.split(',')]

results = []

def process_video(file_path):
    file_tags = set()
    segments, info = whisper_model.transcribe(file_path, beam_size=5)
    text = ''.join(segment.text for segment in segments)
    summary = summarizator(text, min_length = 100, max_length = 2000)
    print(summary)
    '''
    for chunk in chunked_text:
        tags = generate_tags(chunk)
        for t in tags:
            file_tags.add(t)'''
    return None
tag_path = '/home/ubuntu/rutube-hack/tag_list.txt'
rutube_tags = []
with open(tag_path, 'r', encoding='utf-8') as file:
    for line in file:
        rutube_tags.append(line.strip())

for filename in os.listdir(video_path):
    file_path = os.path.join(video_path, filename)
    result = process_video(file_path)
    results.append({'filename': filename, 'tags': result})

# Создаем DataFrame из результатов
df = pd.DataFrame(results)

# Сохраняем DataFrame в CSV файл (по желанию)
df.to_csv('video_processing_results.csv', index=False)

