
import chromadb
import ollama

# Initialize ChromaDB (in-memory database)
# Use PersistentClient() to store data on disk
chromadb_client = chromadb.Client()

# Create a collection to store company policies
collection = chromadb_client.create_collection('company_policies')

# Add company policy documents with unique IDs
collection.add(
    documents=[
        "Employees get 25 days of paid time off (PTO) per year.",
        "Remote work is allowed up to 3 days per week with manager approval.",
        "Expense reports must be submitted within 14 days of travel."
    ],
    ids=[
        "policy_01",
        "policy_02",
        "policy_03"
    ]
)

# Ask a question about company policies
user_question = "How many days can I work from home?"

# Search ChromaDB for documents relevant to the question
result = collection.query(
    query_texts=[user_question]
)

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
