import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import pandas as pd

# Import functions from main.py
from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.llms.openrouter import OpenRouter

# Load environment variables
load_dotenv()


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF file."""
    pdf_reader = PDFReader()
    documents = pdf_reader.load_data(file=Path(pdf_path))
    
    # Combine text from all pages
    full_text = ""
    for doc in documents:
        full_text += doc.text + "\n\n"
    
    return full_text


# Function to extract invoice data using LLM
def extract_invoice_data(pdf_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Initialize LLM
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API key not found. Please set it in the .env file.")
        return None
        
    llm = OpenRouter(
        api_key=OPENROUTER_API_KEY,
        max_tokens=2048,
        context_window=8192,
        model="openai/gpt-4o"
    )
    
    # Create prompt for LLM
    prompt = f"""
    Extract the following information from the invoice text and return it as JSON:
    1. The invoice ID (a unique identifier, often a number)
    2. The date of the invoice
    3. A list of line items, each with:
       - Item name
       - Total price

    Return the data in this format:
    {{"invoice_id": "the_extracted_id",
     "date": "YYYY-MM-DD",
     "line_items": [
        {{"item_name": "First item", "total_price": 100.0}},
        {{"item_name": "Second item", "total_price": 200.0}}
      ]
    }}

    Text:
    {text}
    """
    
    # Get response from LLM
    with st.spinner("Analyzing invoice with AI..."):
        response = llm.complete(prompt)
        
    # Parse JSON response
    try:
        # Extract JSON from the response (it might be wrapped in markdown code blocks)
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        invoice_data = json.loads(response_text.strip())
        return invoice_data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing LLM response: {e}")
        st.text(response.text)
        return None


# Main app UI
st.set_page_config(page_title="Drive Command Splitter", layout="wide")
st.title("Drive Command Splitter")
st.write("Upload your invoice PDF to split expenses between you and your partner.")

# Session state initialization
if 'invoice_data' not in st.session_state:
    st.session_state.invoice_data = None
if 'person1_items' not in st.session_state:
    st.session_state.person1_items = []
if 'person2_items' not in st.session_state:
    st.session_state.person2_items = []

# File uploader
uploaded_file = st.file_uploader("Upload your invoice PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    # Process button
    if st.button("Process Invoice"):
        # Extract data from PDF using LLM
        st.session_state.invoice_data = extract_invoice_data(pdf_path)
        
        # Reset selections when processing a new invoice
        st.session_state.person1_items = []
        st.session_state.person2_items = []
        
        # Clean up temporary file
        os.unlink(pdf_path)

# Display invoice data and expense splitting interface
if st.session_state.invoice_data:
    invoice_data = st.session_state.invoice_data
    
    # Display invoice metadata
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Invoice ID:** {invoice_data.get('invoice_id', 'N/A')}")
    with col2:
        st.write(f"**Date:** {invoice_data.get('date', 'N/A')}")
    
    # Create a dataframe for the line items
    if 'line_items' in invoice_data and invoice_data['line_items']:
        # Get names for the two people
        person1_name = st.text_input("Enter your name", value="Person 1")
        person2_name = st.text_input("Enter your partner's name", value="Person 2")
        
        st.write("### Select items for each person")
        st.write(f"Check the boxes to assign items to {person1_name} or {person2_name}.")
        
        # Create a table with checkboxes
        st.write("")
        
        # Table header - remove the CSS that hides checkbox labels
        style_css = """<style>
            .stCheckbox {margin-right: 0;}
            </style>"""
        st.markdown(style_css, unsafe_allow_html=True)
        
        # Create column headers
        header_cols = st.columns([3, 1, 1, 1])
        header_cols[0].write("**Item**")
        header_cols[1].write("**Price (€)**")
        header_cols[2].write(f"**{person1_name}**")
        header_cols[3].write(f"**{person2_name}**")
        
        # Table rows
        for i, item in enumerate(invoice_data['line_items']):
            item_name = item.get('item_name', 'Unknown item')
            price = item.get('total_price', 0.0)
            
            # Create a new set of columns for each row
            row_cols = st.columns([3, 1, 1, 1])
            row_cols[0].write(item_name)
            row_cols[1].write(f"{price:.2f}")
            
            # Checkboxes for each person - use shorter labels
            person1_checked = row_cols[2].checkbox(
                "✓",  # Simple checkmark as label
                key=f"p1_{i}",
                value=i in st.session_state.person1_items
            )
            
            person2_checked = row_cols[3].checkbox(
                "✓",  # Simple checkmark as label
                key=f"p2_{i}",
                value=i in st.session_state.person2_items
            )
            
            # Update session state based on checkbox selections
            if person1_checked and i not in st.session_state.person1_items:
                st.session_state.person1_items.append(i)
            elif not person1_checked and i in st.session_state.person1_items:
                st.session_state.person1_items.remove(i)
                
            if person2_checked and i not in st.session_state.person2_items:
                st.session_state.person2_items.append(i)
            elif not person2_checked and i in st.session_state.person2_items:
                st.session_state.person2_items.remove(i)
        
        # Calculate totals
        person1_total = sum(invoice_data['line_items'][i]['total_price'] 
                           for i in st.session_state.person1_items)
        person2_total = sum(invoice_data['line_items'][i]['total_price'] 
                           for i in st.session_state.person2_items)
        
        # Display totals
        st.write("---")
        st.write("### Expense Summary")
        total_cols = st.columns(2)
        with total_cols[0]:
            st.metric(f"{person1_name}'s Total", f"{person1_total:.2f} €")
        with total_cols[1]:
            st.metric(f"{person2_name}'s Total", f"{person2_total:.2f} €")
        
        # Calculate who owes whom
        total_expense = person1_total + person2_total
        split_amount = total_expense / 2
        
        st.write("---")
        st.write("### Payment Summary")
        st.write(f"**Total Expense:** {total_expense:.2f} €")
        st.write(f"**Equal Split Amount:** {split_amount:.2f} €")
        
        if person1_total > person2_total:
            amount_owed = person1_total - split_amount
            st.write(f"**{person2_name} owes {person1_name}:** {amount_owed:.2f} €")
        elif person2_total > person1_total:
            amount_owed = person2_total - split_amount
            st.write(f"**{person1_name} owes {person2_name}:** {amount_owed:.2f} €")
        else:
            st.write("**All expenses are perfectly split!**")
    else:
        st.error("No line items found in the invoice.")