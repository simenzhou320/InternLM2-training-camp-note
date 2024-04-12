import os
import json
from PyPDF2 import PdfReader
import threading
import re

def pdf_to_jsonl(pdf_path, jsonl_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)

        data = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text:
                # Remove newlines and spaces
                text = re.sub(r'\s+|徐文兵：|梁冬：', '', text)
                
                if len(text) == 0:
                    continue

                # Split text by Chinese period
                sentences = re.split(r'。', text)
                
                # Limit the length of each sentence to 50 characters
                sentences = [s[:50] for s in sentences]
                
                for s in sentences:
                    if len(s) == 0:
                        continue
                    data.append({"conversation": [{"input": "", "output": s}]})

                #data.append({"conversation": [{"system": "", "input": "", "output": s}] for s in sentences})

    with open(jsonl_path, 'w') as file:
        file.write('[')
        for item in data:
            json.dump(item, file, ensure_ascii=False, indent=4)
            file.write(',\n')
        file.write(']')

def convert_pdf_to_jsonl_thread(pdf_path, jsonl_path, thread_index):
    print(f"Thread {thread_index}: Start converting {pdf_path}")
    pdf_to_jsonl(pdf_path, jsonl_path)
    print(f"Thread {thread_index}: Finish converting {pdf_path}")

def convert_pdfs_in_directory(pdf_directory, output_directory, num_threads):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    pdf_paths = [os.path.join(pdf_directory, f) for f in pdf_files]
    jsonl_paths = [os.path.join(output_directory, f"{os.path.splitext(f)[0]}.jsonl") for f in pdf_files]

    threads = []
    for i, (pdf_path, jsonl_path) in enumerate(zip(pdf_paths, jsonl_paths)):
        if len(threads) == num_threads:
            # Wait for any of the threads to finish before starting a new one
            threads[0].join()
            threads.pop(0)

        thread = threading.Thread(target=convert_pdf_to_jsonl_thread, args=(pdf_path, jsonl_path, i))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    pdf_directory = 'book'
    output_directory = 'output'
    num_threads = 5
    convert_pdfs_in_directory(pdf_directory, output_directory, num_threads)

