from openai import OpenAI
class LLM():
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(
            api_key = self.api_key,
            base_url= 'https://api.vsegpt.ru/v1'
        )
    def generate_tags(self, audio_chunk, target_tags):
        system_prompt = f"Сгенерируй список тегов по описанию видео, которое отправит пользователь. Выведи только самые подходящие, обобщенные теги, сильно не конкретизируй, не додумывай, желательно 1-3 тега. В случае если тебе не нравится текст по любым причинам, ответь пустой строкой. Теги должны быть разделены запятой. Вот название и описание видео, чтобы ты знал контекст: {target_tags}"
        prompt = audio_chunk
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
        except Exception as e:
            print(e)
            return []
        
