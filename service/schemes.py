from pydantic import BaseModel
from datetime import datetime
from typing import List

class TranscribationItem(BaseModel):
    id_audio_transcribation: int
    audio_id: int
    transcribation: str
    start_interval: datetime
    end_interval: datetime
    id_action: int

class TranscribationRequest(BaseModel):
    transcriptions: List[TranscribationItem]