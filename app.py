import streamlit as st
import PyPDF2
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

from stqdm import stqdm  
import torch
import textwrap
from dotenv import load_dotenv
import os
def configure():
    load_dotenv()
def prompt_formatter(query: str,
                     context_items: list[dict],tokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What is the Stoic concept of 'apatheia'?
Answer: The Stoic concept of 'apatheia' refers to a state of mind where one is free from unhealthy passions or disturbances, such as excessive desires, fears, or anxieties. It does not mean total emotional detachment, but rather a calm and balanced state of being where one's emotions are under rational control. Apatheia allows individuals to respond to external events with clarity and equanimity, without being overwhelmed by emotional reactions.
\nExample 2:
Query: How did Stoicism influence Roman philosophy and society?
Answer: Stoicism had a profound influence on Roman philosophy and society, particularly during the Imperial period. Roman Stoics such as Seneca, Epictetus, and Marcus Aurelius emphasized principles of virtue, self-discipline, and resilience in the face of adversity. Stoic teachings were integrated into Roman educational systems, legal theory, and political discourse, shaping the moral character of individuals and the governance of the empire. The Stoic emphasis on duty, justice, and natural law contributed to the development of Roman law and ethics.
\nExample 3:
Query: What are some Stoic practices for achieving tranquility?
Answer: Stoics employ various practices to cultivate tranquility and inner peace, such as negative visualization, mindfulness of the present moment, and voluntary discomfort. Negative visualization involves contemplating the loss of things we value, which helps to appreciate them more fully and reduce attachment. Mindfulness helps individuals focus on what is within their control and accept the present moment without judgment. Voluntary discomfort, such as fasting or exposure to cold, builds resilience and strengthens the willpower to endure hardship.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt
def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()  
    return cleaned_text

def open_and_read_pdf(pdf) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    with pdf as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages_and_texts = []
        for i, page_obj in enumerate(pdf_reader.pages, start=1):
            text = page_obj.extract_text()
            text = text_formatter(text)
            pages_and_texts.append({
                "page_number": i,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                "text": text
            })
    return pages_and_texts

def pages_chunks(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks = []
    for item in pages_and_texts:
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """Splits the input_list into sublists of size slice_size (or as close as possible)."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def elimination_chunks(df: pd.DataFrame, min_token_length) -> list[dict]:
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

    return pages_and_chunks_over_min_token_len


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                embedding_model: SentenceTransformer,
                                n_resources_to_return: int = 5):
    """
    Embeds a query with the model and returns top K scores and indices from embeddings.
    """
    # Embed the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    
    # Get the number of scores above a threshold (e.g., 0.5)
    valid_scores = dot_scores > 0.5
    cnt = valid_scores.sum().item()
    
    # Get top K scores and their indices
    top_k = min(cnt, n_resources_to_return)
    scores, indices = torch.topk(dot_scores, k=top_k)
    
    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 embedding_model:SentenceTransformer,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  embedding_model=embedding_model,
                                                  n_resources_to_return=n_resources_to_return)

    st.write(f"Query: {query}\n")
    st.write("Results:")
    context_items=[]
    for score, index in zip(scores, indices):
        context_items.append(pages_and_chunks[index])
        st.write(f"**Score:** {score:.4f}")
        st.write(f"**Sentence Chunk:** {pages_and_chunks[index]['sentence_chunk']}")
        st.write(f"**Page Number:** {pages_and_chunks[index]['page_number']}")
        st.write("\n")
    return context_items
def ask(query, model, embedding_model, embeddings, pages_and_chunks, tokenizer,
        temperature=0.7, max_new_tokens=512, format_answer_text=True, return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    print("Starting ask function")
    print(f"Query: {query}")
    
    # Get just the scores and indices of top related results
    print("Retrieving relevant resources")
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings, embedding_model=embedding_model)
    print(f"Scores: {scores}")
    print(f"Indices: {indices}")

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()  # return score back to CPU

    print(f"Context items: {context_items}")

    # Format the prompt with context items
    print("Formatting the prompt")
    prompt = prompt_formatter(query=query, context_items=context_items, tokenizer=tokenizer)
    print(f"Prompt: {prompt}")

    # Tokenize the prompt
    print("Tokenizing the prompt")
    input_ids = tokenizer(prompt, return_tensors="pt").to("cpu")
    print(f"Input IDs: {input_ids}")

    # Generate an output of tokens
    print("Generating output tokens")
    outputs = model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    print(f"Output tokens: {outputs}")

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])
    print(f"Output text before formatting: {output_text}")

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")
        print(f"Output text after formatting: {output_text}")

    # Only return the answer without the context items
    if return_answer_only:
        print("Returning answer only")
        return output_text

    print("Returning answer with context items")
    return output_text, context_items


with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    ''')
    st.markdown('''
    ## Instructions
    1. Upload a PDF file.
    2. Ask questions about the content of your PDF.
    ''')

def main():
    configure()
    st.header("Chat with PDF 💬")
    
    MAX_UPLOAD_SIZE_MB = 30
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    pdf = st.file_uploader(f"Upload your PDF (Limit {MAX_UPLOAD_SIZE_MB}MB per file)", type='pdf')
    query = st.text_input("Ask questions about your PDF file:")
    
    if pdf:
        if pdf.size > MAX_UPLOAD_SIZE_BYTES:
            st.error(f"File size is too large! Please upload a file smaller than {MAX_UPLOAD_SIZE_MB} MB.")
            return
        
        with st.spinner('Processing PDF...'):
            pages_and_texts = open_and_read_pdf(pdf)

        nlp = English()
        nlp.add_pipe("sentencizer")
        
        for item in stqdm(pages_and_texts, desc="Tokenizing pages"):
            item["sentences"] = list(nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])
        
        df = pd.DataFrame(pages_and_texts)
        sent = df['page_sentence_count_spacy'].describe().round(2)['mean']
        token = df['page_token_count'].describe().round(2)['mean']
        slice_size = round((340 * sent) / token)
        
        for item in stqdm(pages_and_texts, desc="Splitting sentences into chunks"):
            item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=slice_size)
            item["num_chunks"] = len(item["sentence_chunks"])
        
        pages_and_chunks = pages_chunks(pages_and_texts)
        df = pd.DataFrame(pages_and_chunks)
        pages_and_chunks = elimination_chunks(df, 30)
        
        text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")            

        if query:
            with st.spinner('Generating response...'):
                embeddings = embedding_model.encode(text_chunks, batch_size=64, convert_to_tensor=True)
                '''print_top_results_and_scores(query=query,pages_and_chunks=pages_and_chunks,embedding_model=embedding_model,
                             embeddings=embeddings)'''
                #importing the model 
                model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", 
                        device_map="cpu", 
                        torch_dtype="auto", 
                        trust_remote_code=True, 
                        token='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'
                        )
                print(model)
                tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",token='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')
                answer, context_items =ask(query,model,embedding_model,embeddings,pages_and_chunks,tokenizer,
                    temperature=0.7,
                    max_new_tokens=512,
                    format_answer_text=True,
                    return_answer_only=True)
            st.text(answer)

if __name__ == "__main__":
    main()
