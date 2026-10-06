import pymupdf
from pathlib import Path

def chunking(pdf_path, paper_name, chunk_size=1000, overlap=200):
    document = pymupdf.open(pdf_path)
    chunk_data_list = []
    
    for page_number, page in enumerate(document):
        page_text = page.get_text()
        for i in range(0, len(page_text), chunk_size-overlap):
            chunk = page_text[i:i+chunk_size]
            
            chunk_data = {
            "text": chunk,
            "paper": paper_name,
            "page": page_number + 1
            }
            
            chunk_data_list.append(chunk_data)
    
    return chunk_data_list

def load_corpus(folder):
    data_folder = Path(folder)
    all_chunks = []
    for pdf_path in data_folder.glob("*.pdf"):
        all_chunks.extend(chunking(pdf_path, pdf_path.stem))
    return all_chunks


def create_embeddings(all_chunks, model):
    chunk_text_list = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(chunk_text_list)
    return embeddings

