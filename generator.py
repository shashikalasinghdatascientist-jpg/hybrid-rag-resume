import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

class EvidenceBasedGenerator:
    def __init__(self):
        # Initialize Claude 3.5 Sonnet
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a strict evidence-based AI. Your goal is to answer the question "
                "using ONLY the provided context. \n\n"
                "CRITICAL RULES:\n"
                "1. If the answer is not in the context, state 'Information not found'.\n"
                "2. For every claim in your answer, you must provide a verbatim quote from the context.\n"
                "3. The quotes MUST be exact substrings of the context provided.\n\n"
                "Response Format (JSON):\n"
                "{{\n"
                "  \"answer\": \"The final answer to the question\",\n"
                "  \"evidence\": [\"exact quote 1\", \"exact quote 2\"]\n"
                "}}"
            )),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

        self.chain = self.prompt | self.llm | JsonOutputParser()

    def generate(self, query, context_chunks):
        context_text = "\n---\n".join(context_chunks)
        try:
            response = self.chain.invoke({"context": context_text, "question": query})
            return response
        except Exception as e:
            return {"answer": f"Error generating response: {str(e)}", "evidence": []}

    def validate_evidence(self, response, context_chunks):
        """Checks if every quote in the response actually exists in the context."""
        evidence = response.get("evidence", [])
        all_context = " ".join(context_chunks)

        validated = []
        missing = []

        for quote in evidence:
            if quote and quote in all_context:
                validated.append(quote)
            else:
                missing.append(quote)

        return {
            "is_fully_validated": len(missing) == 0,
            "validated_quotes": validated,
            "missing_quotes": missing
        }

if __name__ == "__main__":
    # Test run
    gen = EvidenceBasedGenerator()
    test_context = [
        "AI Engineer with 7+ years of experience delivering machine learning, NLP and Generative AI systems.",
        "Skilled in Python, FastAPI, LangChain, LangGraph."
    ]
    test_query = "How many years of experience does the candidate have?"

    res = gen.generate(test_query, test_context)
    val = gen.validate_evidence(res, test_context)

    print(f"Response: {json.dumps(res, indent=2)}")
    print(f"Validation: {json.dumps(val, indent=2)}")
