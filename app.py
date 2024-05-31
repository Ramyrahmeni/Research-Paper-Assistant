import streamlit as st
import PyPDF2
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer, util
from stqdm import stqdm  # for progress bars in Streamlit
import torch
import textwrap

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()  # note: this might be different for each doc (best to experiment)
    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
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
                                model: SentenceTransformer,
                                n_resources_to_return: int=5,
                                ):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query,
                                   convert_to_tensor=True)

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    
    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

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

    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")

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
                #performing vector search
                print_top_results_and_scores(query=query,pages_and_chunks=pages_and_chunks,embedding_model=embedding_model,
                             embeddings=embeddings)

if __name__ == "__main__":
    main()
