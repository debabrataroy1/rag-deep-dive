
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB (in-memory database)
# Use PersistentClient() to store data on disk
chromadb_client = chromadb.Client()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a collection to store company policies
collection = chromadb_client.create_collection('company_policies')

# Add company policy documents with unique IDs

documents=[
        "Employees get 25 days of paid time off (PTO) per year.",
        "Remote work is allowed up to 3 days per week with manager approval.",
        "Expense reports must be submitted within 14 days of travel."
    ]
documents_embeding = embedding_model.encode(documents).tolist()

ids=[
        "policy_01",
        "policy_02",
        "policy_03"
    ]
collection.upsert(ids=ids, embeddings=documents_embeding, documents=documents)

# Ask a question about company policies
user_question = "How many days can I work from home?"

embedding_question = embedding_model.encode(user_question)

# Search ChromaDB for documents relevant to the question
result = collection.query(embedding_question)

# Create a prompt using the retrieved documents and user question
prompt = f"""
Answer the question using only the context below.

Context: {result["documents"]}

Question: {user_question}
"""

# Generate an answer using the local Llama 3.2 model
response = ollama.generate(
    'llama3.2:latest',
    prompt=prompt
)

# Print the generated answer
print(response["response"])
