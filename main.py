from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
import json

load_dotenv()

class LineItem(BaseModel):
    """A line item in an invoice."""
    
    item_name: str = Field(description="The name of this item")
    total_price: float = Field(description="The total price of this item")

class Invoice(BaseModel):
    """A representation of information from an invoice."""
    
    invoice_id: str = Field(description="A unique identifier for this invoice, often a number")
    date: datetime = Field(description="The date this invoice was created")
    line_items: List[LineItem] = Field(description="A list of all the items in this invoice")

# Loading the invoice, will be replaced by the body of the request
from llama_index.readers.file import PDFReader
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF file.
    
    For very large PDFs, this function will paginate the content to ensure
    it fits within the LLM's context window.
    """
    pdf_reader = PDFReader()
    documents = pdf_reader.load_data(file=Path(pdf_path))
    
    print(f"PDF loaded successfully. Found {len(documents)} pages.")
    
    # Combine text from all pages
    full_text = ""
    for i, doc in enumerate(documents):
        page_text = doc.text.strip()
        full_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
    
    # Check if the text is too large and needs pagination
    # Rough estimate: 1 token ≈ 4 characters
    estimated_tokens = len(full_text) / 4
    max_safe_tokens = 6000  # Leave room for prompt and response
    
    if estimated_tokens > max_safe_tokens:
        print(f"Warning: PDF content is large (est. {estimated_tokens:.0f} tokens). Processing may be limited.")
    
    return full_text

# Extract text from all pages of the PDF
pdf_path = "./bon_commande.pdf"
text = extract_text_from_pdf(pdf_path)

# Calling the LLM
from llama_index.llms.openrouter import OpenRouter

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    max_tokens=2048,  # Increased to handle larger responses
    context_window=8192,  # Increased context window
    # model="google/gemini-2.5-flash",
    model="openai/gpt-4o"
)

# Use regular completion instead of structured output to avoid Pydantic compatibility issues
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

response = llm.complete(prompt)
print(response.text)