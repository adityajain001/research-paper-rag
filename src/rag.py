from sklearn.metrics.pairwise import cosine_similarity
import requests

def retrieve(query, model, embeddings, k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    scores = similarities[0]
    top_indices = scores.argsort()[-k:][::-1]
    return top_indices


def build_context(top_indices, all_chunks):
    context = ""
    for index in top_indices:
        context += f"[{all_chunks[index]['paper']}, Page {all_chunks[index]['page']}]\n"
        context += all_chunks[index]["text"]
        context += "\n\n"
    return context


def build_prompt(context, query):
    return f"""
    You are answering questions using a collection of research papers.

    Use only the context provided below to answer the question.
    If the answer is not contained in the context, say that you cannot answer it from the provided context.
    Cite the relevant paper name and page number(s) in your answer.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """




def generate_answer(prompt, llm_model, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": llm_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
    ]
    }

    response = requests.post(url, headers=headers, json=data)
    
    status_code = response.status_code
    result = response.json()

    if status_code != 200:
        return status_code, None, result
    
    model_name = result["model"]
    answer = result["choices"][0]["message"]["content"]
    return status_code, model_name, answer
