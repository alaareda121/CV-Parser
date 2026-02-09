import streamlit as st
import PyPDF2
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# -----------------------
# 1) Define JSON Schema
# -----------------------
class CVSchema(BaseModel):
    full_name: str = Field(description="Candidate full name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    summary: str = Field(description="Professional summary")
    education: list = Field(description="Education list")
    skills: list = Field(description="Skills list")
    experience: list = Field(description="Work experience")
    projects: list = Field(description="Projects list")

parser = JsonOutputParser(pydantic_object=CVSchema)
format_instructions = parser.get_format_instructions()

# -----------------------
# 2) Load Model
# -----------------------
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------
# 3) Extract Text from PDF
# -----------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -----------------------
# 4) LLM Generator
# -----------------------
def generate_text(prompt, max_length=1500):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------
# 5) Extract JSON Block
# -----------------------
def extract_json_block(text):
    pattern = r"```json\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

# -----------------------
# 6) Streamlit GUI
# -----------------------
st.title("CV Parser (PDF → JSON)")
st.write("Upload a CV in PDF format and extract structured information using AI.")

uploaded_pdf = st.file_uploader("Upload CV (PDF)", type=["pdf"])

if uploaded_pdf:
    if st.button("Analyze CV"):
        with st.spinner("Reading PDF..."):
            cv_text = extract_text_from_pdf(uploaded_pdf)

        st.write("### Extracted Text:")
        st.text(cv_text[:800] + " ...")

        # Prompt
        prompt = f"""
You are a CV parsing assistant.
Extract structured data from this CV.

{format_instructions}

CV TEXT:
{cv_text}
"""

        with st.spinner("Analyzing CV using AI model..."):
            raw_output = generate_text(prompt)

        st.write("### Raw Output:")
        st.code(raw_output)

        json_block = extract_json_block(raw_output)

        if json_block:
            try:
                parsed_data = parser.parse(json_block)
                st.success("Structured JSON extracted:")
                st.json(parsed_data)
            except Exception as e:
                st.error(f"Error parsing JSON: {e}")
        else:
            st.error("No JSON block found in model output.")

