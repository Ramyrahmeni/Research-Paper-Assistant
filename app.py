import streamlit as st
import PyPDF2
import pandas as pd
from spacy.lang.en import English # see https://spacy.io/usage for install instructions
import re
from sentence_transformers import SentenceTransformer
def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf: __file__) -> list[dict]:
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

            # Display the number of pages in the PDF file
            #print(f"Number of pages: {len(pdf_reader.pages)}")

            # Extract text from page 0 (optional)
            page_obj = pdf_reader.pages[0]
            #print(len(page_obj.extract_text()))  # open a document
            pages_and_texts = []
            i=1
            for page_obj in pdf_reader.pages:  # iterate the document pages
                text = page_obj.extract_text()  # get plain text encoded as UTF-8
                text = text_formatter(text)
                #print(i)
                #print(text)
                
                pages_and_texts.append({"page_number": i ,  # adjust page numbers since our PDF starts on page 42
                                            "page_char_count": len(text),
                                            "page_word_count": len(text.split(" ")),
                                            "page_sentence_count_raw": len(text.split(". ")),
                                            "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                            "text": text})
                i+=1
            return pages_and_texts
def pages_chunks(pages_and_texts:list[dict])->list[dict]:
    pages_and_chunks = []
    for item in pages_and_texts:
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks
        

def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
def elimination_chunks(df:pd.DataFrame,min_token_length) -> list[dict]:
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    return pages_and_chunks_over_min_token_len

    
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    ''')
def main():
    st.header("Chat with PDF 💬")
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf:
        pages_and_texts = open_and_read_pdf(pdf)
        #print(pages_and_texts[:2])

        #Lets Split Pages into sentences
        nlp = English()

        nlp.add_pipe("sentencizer")
        for item in pages_and_texts:
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_count_spacy"] = len(item["sentences"])

        #Lets chunk our sentences

        df = pd.DataFrame(pages_and_texts)
        sent = df['page_sentence_count_spacy'].describe().round(2)['mean']
        token= df['page_token_count'].describe().round(2)['mean']
        print('sentence mean',sent)
        print('token mean',token)
        slice_size=round((340*sent)/token)
        #print(slice_size)
        for item in pages_and_texts:
            item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                slice_size=slice_size)
            item["num_chunks"] = len(item["sentence_chunks"])
        
        #Chunking our sentences 
        pages_and_chunks=pages_chunks(pages_and_texts)

        #eliminating_short_chunks
        df=pd.DataFrame(pages_and_chunks)
        pages_and_chunks=elimination_chunks(df,30)
        #pages_and_chunks[:2]

        #embedding_chunks

        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")
        sentences = [
            "The Sentences Transformers library provides an easy and open-source way to create embeddings.",
            "Sentences can be embedded one by one or as a list of strings.",
            "Embeddings are one of the most powerful concepts in machine learning!",
            "Learn to use embeddings well and you'll be well on your way to being an AI engineer."
        ]

        # Sentences are encoded/embedded by calling model.encode()
        embeddings = embedding_model.encode(sentences)
        embeddings_dict = dict(zip(sentences, embeddings))

        # See the embeddings
        for sentence, embedding in embeddings_dict.items():
            print("Sentence:", sentence)
            print("Embedding:", embedding)
            print("")


        
        
        
        
        
    

if __name__ == "__main__":
    main()
