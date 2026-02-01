from pydantic import BaseModel,Field
from typing import Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

load_dotenv()

model =  ChatGroq(model = 'llama-3.1-8b-instant',
    temperature=0,
    max_retries=2)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Write a 2 detailed facts on {topic} {format_instruction}',
    input_variables= ['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

template2 = PromptTemplate(
    template='Write a 5 line summary on following text. /n{text}',
    input_variables= ['text']
)

##propmpt1 = template1.invoke({'topic': 'LLM'})

##result = model.invoke(propmpt1)

##propmpt2 = template2.invoke({'text': result.content})

##result1 = model.invoke(propmpt2)

##print(result1.content)


chain = template1 | model | parser 

result =chain.invoke({'topic': 'Saturn'})

print(result)