import os
from pypdf import PdfReader

def extract_text_from_pdfs(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist.")
            return

        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing {filename}...")
                f_out.write(f"\n\n--- START OF {filename} ---\n\n")
                
                try:
                    reader = PdfReader(file_path)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            f_out.write(f"\n[Page {i+1}]\n{text}")
                except Exception as e:
                    f_out.write(f"\nError processing {filename}: {e}\n")
                
                f_out.write(f"\n\n--- END OF {filename} ---\n\n")

if __name__ == "__main__":
    extract_text_from_pdfs('reaserchpapers', 'pdf_contents.txt')
