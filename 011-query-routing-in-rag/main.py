
from app.rag_pipeline import RAGPipeline


def main():

    print("Initializing RAG System...")

    rag = RAGPipeline()

    print("\nRAG System is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting RAG System...")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            result = rag.ask(question)

            print("\nAnswer:")
            print(result)

            print("\n" + "-" * 60 + "\n")

        except Exception as e:

            print(f"Error: {e}")


if __name__ == "__main__":
    main()
