
from pypdf import PdfReader
import os

file_path = r"c:\Users\prade\Desktop\ml_minor\reaserchpapers\research_paper (template).pdf"

print(f"Checking file: {file_path}")
if not os.path.exists(file_path):
    print("File not found!")
else:
    try:
        reader = PdfReader(file_path)
        print(f"Number of pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"--- Page {i+1} ---")
            print(text[:500]) # Print first 500 chars
            if not text:
                print("[No text extracted]")
    except Exception as e:
        print(f"Error: {e}")
