from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts  import PromptTemplate
load_dotenv()

model = ChatGroq(model = 'llama-3.1-8b-instant',
    temperature=1.5,
    max_retries=2)


st.header("Reseach tool")

#drop down
paper_input = st.selectbox("Select option for name ",["BERT: pre-training of deep learning  bidirectional transformers","microsoft","Caddlac","Doll house"])
style_input = st.selectbox("Select option for style",["i","Mathematical","U","N"])
length_input = st.selectbox("Select option for length ",["1","2 sentence","3","4"])

##  user_input = st.text_input("enter your prompopt")


#templater
template = PromptTemplate(
    template= """ Please summarize the research paper titled "{paper_input}" with the following  specifications:  Explanation Style: {style_input} Explanation Length: {length_input}"
    "1. Mathematical Details: - Include relevant mathematical equations if present in the paper. - Explain the mathematical concepts using simple, intuitive code snippets where applicable. "
    "2. Analogies: - Use relatable analogies to simplify complex ideas. "
    "If certain information is not available in the paper, respond with: " Insufficient 
information available" instead of guessing. 
Ensure the summary is clear, accurate, and aligned with the provided style and 
length.""",
input_variables = ['paper_input', 'style_input', 'length_input']
)

#fill placeholeders

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

##using chain

chain = template | model
chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})


if st.button('Summerrize'):
    result = model.invoke(prompt)
    st.write(result.content)