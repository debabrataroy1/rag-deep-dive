
import chromadb
import ollama
from strategies import semantic_similarity_chunking
from pypdf import PdfReader
from pathlib import Path
from sentence_transformers import SentenceTransformer


def init_resource():
    # Initialize ChromaDB PersistentClient() to store data on disk
    chromadb_client = chromadb.PersistentClient('./chroma_db')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Create a collection to store company policies
    collection = chromadb_client.get_or_create_collection('book')
    
    return embedding_model, collection

def load_pdf():
    pdf_path = Path(__file__).parent / "java_notes.pdf"
    return PdfReader(pdf_path)

def extract_text(reader: PdfReader):
    page_text = []
    for page in reader.pages:
        if text := page.extract_text():
            page_text.append(text)
        
    return '\n'.join(page_text)


    
def embedding(chunks):
    return embedding_model.encode(chunks)
    
def save_chunks_in_db(chunks, embedded_chunks):
    ids = [f"text {i}" for i in range(len(embedded_chunks))]
    
    batch_size = 2000
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = chunks[i : i + batch_size]
        batch_embeddings = embedded_chunks[i : i + batch_size]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
  )

    
    # collection.upsert(ids=ids,
    #                   embeddings = embedded_chunks,
    #                   documents = chunks)
    

def retrive_data_from_db(question):
    return collection.query(question, include=['documents'])


def build_context(results):
    context = []
    for index, data in enumerate(results):
        documents = results['documents']
        context.append(f"source {index}:{documents}")
    return '\n\n'.join(context)

def create_prompt(context, query):
    return f""" Answer the question using only the context below.
Context: {context}

Question: {query}
"""

embedding_model, collection = init_resource()
pdf_data = load_pdf()
pdf_text = extract_text(pdf_data)
chunks = semantic_similarity_chunking(pdf_text)
embedded_chunks = embedding(chunks)
save_chunks_in_db(chunks, embedded_chunks)

# Ask a question 
user_question = "what is OBJECT-ORIENTED Programming?"

embedding_question = embedding(user_question)

# Search ChromaDB for documents relevant to the question
result = retrive_data_from_db(embedding_question)

context = build_context(result)

# Create a prompt using the retrieved documents and user question
prompt = create_prompt(context, user_question)

# Generate an answer using the local Llama 3.2 model
response = ollama.generate(
    'llama3.2:latest',
    prompt=prompt
)

# Print the generated answer
print(response["response"])
