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
        async with httpx.AsyncClient() as client:
            response = await client.post(url, file=temp_file_path)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error from transcription service: {response.text}")
        
        transcription_data = response.json()

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