from openai import OpenAI
class LLM():
    def __init__(self):
        self.api_key = 'sk-or-vv-ed9b0360080c7f2ebfea7bb6d596cc6f67d2d2d8356b6cd950e66379b63c2d87'
        self.client = OpenAI(
            api_key = self.api_key,
            # base_url = 
        )
    def generate_tags(self, audio_chunk, video_chunk, target_tags):
        system_prompt = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, обобщенные теги, сильно не конкретизируй, не додумывай, желательно 1-3 тега. В случае если тебе не нравится текст по любым причинам, ответь пустой строкой. Теги должны быть разделены запятой. Вот название и описание видео, чтобы ты знал контекст: {target_tags}"
        prompt = audio_chunk + '\n' + video_chunk
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
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
        except:
            return []
        
