import os
from retriever import HybridRetriever
from generator import EvidenceBasedGenerator
from dotenv import load_dotenv

load_dotenv()

def main():
    print("--- Hybrid Evidence-Based RAG System ---")

    # 1. Initialize components
    retriever = HybridRetriever()
    generator = EvidenceBasedGenerator()

    while True:
        query = input("\nAsk a question about the resume (or 'exit' to stop): ")
        if query.lower() in ['exit', 'quit']:
            break

        print("\nRetrieving evidence...")
        context_chunks = retriever.retrieve(query)

        print("Generating evidence-based answer...")
        response = generator.generate(query, context_chunks)

        print("Validating evidence...")
        validation = generator.validate_evidence(response, context_chunks)

        print("\n" + "="*50)
        print(f"QUESTION: {query}")
        print(f"ANSWER: {response.get('answer')}")
        print(f"VALIDATION: {'✅ FULLY VALIDATED' if validation['is_fully_validated'] else '❌ VALIDATION FAILED'}")

        if validation['is_fully_validated']:
            print("\nEvidence used:")
            for quote in response.get('evidence', []):
                print(f"- \"{quote}\"")
        else:
            print("\nMissing/Incorrect Evidence:")
            for quote in validation['missing_quotes']:
                print(f"- {quote}")
        print("="*50)

if __name__ == "__main__":
    main()
