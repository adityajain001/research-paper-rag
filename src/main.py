from ingest import load_corpus, create_embeddings
from rag import retrieve, build_context, build_prompt, generate_answer
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = load_corpus('data')
    embeddings = create_embeddings(all_chunks, model)

    query = input('Ask a question: ')

    top_indices = retrieve(query,model,embeddings)

    context = build_context(top_indices, all_chunks)

    prompt = build_prompt(context, query)

    llm_model = "nvidia/nemotron-3-super-120b-a12b:free"
    #llm_mode = "openrouter/free" #if nividia stops functioning

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    status_code, model_name, answer = generate_answer(prompt, llm_model, api_key)

    print(f"Status Code: {status_code}")
    print(f"Model name: {model_name}")
    print("Answer:")
    print(answer)
