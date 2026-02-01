from pydantic import BaseModel,Field
from typing import Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

model =  ChatGroq(model = 'llama-3.1-8b-instant',
    temperature=0,
    max_retries=2)
# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Job Skill extractor", page_icon="🤖")
st.title("🤖 Skill extractor")


class Extract(BaseModel):
    key_words: list[str] = Field(description = "List of 5-10 key skills/technologies from the job description")
    summary: str = Field(description = " A brief summary of job description in 5 bullet points")
    level: Optional[str] = Field(default= 'Entry level', description= "return level of experience required. Either Entry level, mid-level or senior level")
    name: str = Field(description="write down name of company")


job_summary =st.text_area("Paste the job description below:", height=250, placeholder="e.g., We are looking for a Senior Software Engineer with 5+ years of Python, AWS, and Kubernetes experience at Acme Corp...")
if st.button("Extract Skills"):
    if not job_summary.strip():
        st.warning("Please paste a job description first.")
    else:
        structured_model = model.with_structured_output(Extract)
        result = structured_model.invoke(job_summary)

        # --- Display results in a user-friendly format ---
        st.subheader("🧠 Extracted Information")
        st.markdown("---")

        col1, col2 = st.columns(2)
                
        with col1:
                    st.metric(label="🏢 Company Name", value=result.name)
        with col2:
                    st.metric(label="📈 Experience Level", value=result.level)
                
        st.info(f"**Job Summary:** {result.summary}")
                
        st.subheader("🛠️ Key Skills/Technologies")
        st.caption(f"Found **{len(result.key_words)}** skills.")

        
        skills_html = "".join(
                    [f'<span style="display:inline-block; background-color:#1E90FF; color:white; border-radius:12px; padding:4px 10px; margin:2px; font-size:0.9em;">{skill}</span>' 
                     for skill in result.key_words]
                )
        st.markdown(skills_html, unsafe_allow_html=True)

