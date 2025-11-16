import pandas as pd
import requests

class GroqChatAgent:
    def __init__(self, api_key, model="llama-3.3-70b-versatile", dataframe_path="products.json"):
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = model
        self.df = pd.read_json(dataframe_path)

    def query_groq(self, user_input):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Extract some product info to add to the prompt
        # Here we take first 5 products' name and description as an example
        product_snippet = self.df.head(5).to_dict(orient="records")
        product_info = "\n".join([f"{p['brand']} {p['name']}: {p.get('description', 'No description')}" for p in product_snippet])

        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant knowledgeable about the products."},
            {"role": "system", "content": f"Here are some products:\n{product_info}"},
            {"role": "user", "content": user_input}
        ]

        payload = {
            "model": self.model,
            "messages": prompt_messages,
            "max_tokens": 150,
            "temperature": 0.7
        }

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API Error {response.status_code}: {response.text}")

    def ask(self, question):
        return self.query_groq(question)

def load_agent(api_key):
    return GroqChatAgent(api_key)
