from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import time
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from models.audio_pipeline import AudioProcesser
from models.video_pipeline import Caption
import json

app = FastAPI()
audio_processer = AudioProcesser()
video_processer = Caption()

@app.post("/get_tags")
def get_tags(video_description: str, file: UploadFile = File(...)):
    with TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush ()
        tags= audio_processer.process_video(temp_file_path, video_description)
        video_caption = video_processer.shot_transit(temp_file_path)
        data = json.loads(video_caption)
        video_caption =" ".join(item['caption'] for item in data)
        video_tags = audio_processer.generate_tags(video_caption, video_description)
        tags.update(video_tags)
        print(tags)
    try:
        rutube_tags = []
        tag_path = 'tag_list.txt'
        with open(tag_path, 'r', encoding='utf-8') as file:
            for line in file:
                rutube_tags.append(line.strip())
        result_tags = audio_processer.get_nearest_tags(tags, rutube_tags, video_description)
        return result_tags
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/get_tags_for_zip")
def get_tags_for_zip():
    pass