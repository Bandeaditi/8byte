import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from bankend.llm import llm as client
import streamlit as st
from bankend.parse import parse_receipt
from bankend.database import save_receipt, init_db
import tempfile
import os

# Initialize DB
init_db()

st.title("Receipt Processor with LLM")

# File Upload
uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "png", "pdf", "txt"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        receipt = parse_receipt(tmp.name)
        save_receipt(receipt)
        os.unlink(tmp.name)
    
    st.success("Receipt parsed and saved!")
    st.json(receipt.dict())

# Natural Language Query
query = st.text_input("Ask a question about your expenses:")
if query:
    from bankend.llm import client
    from bankend.database import Receipt
    import pandas as pd

    conn = sqlite3.connect("receipts.db")
    df = pd.read_sql("SELECT * FROM receipts", conn)
    conn.close()

    prompt = f"""
    Analyze this expense data and answer the question:
    Data: {df.to_dict()}
    Question: {query}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)