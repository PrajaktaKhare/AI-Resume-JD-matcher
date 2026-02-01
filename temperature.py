from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = 'llama-3.1-8b-instant',
    temperature=1.5,
    max_retries=2)

result = model.invoke("Write a 2 line poem on layoffs")

print(result.content)