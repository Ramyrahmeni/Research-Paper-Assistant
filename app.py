import streamlit as st
import PyPDF2
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM , pipeline
import os
from stqdm import stqdm  
import torch
import textwrap
from dotenv import load_dotenv
import google.generativeai as gen_ai
import gc
import pickle


load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-1.5-pro')
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])          

# Initialize chat session in Streamlit if not already present
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
    valid_scores = dot_scores > 0.1
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
def ask(query, embedding_model, embeddings, pages_and_chunks):
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

    #st.text(f"Context items: {context_items}")

    # Format the prompt with context items
    examples = [
    # Example 1: Project Management
    {
        "context": [
            {
                "page_number": 20,
                "sentence_chunk": "1.5. Adopted Methodology 11 The main elements of Scrum include roles, events, and artifacts. The Scrumban team consists of three roles: the Product Owner, who represents the stakeholders and manages the product backlog. The Scrum Master, who facilitates the Scrum process and supports the team. The Developers, who are responsible for delivering increments of product functionalities. Figure 1.6 shows brief summary of how Scrumban team works."
            },
            {
                "page_number": 21,
                "sentence_chunk": "1.5 Adopted Methodology In defining our project management approach, we have adopted a dual methodology, leveraging both Scrumban and the CRISP-DM framework. Scrumban, a hybrid of Kanban’s adaptability and Scrum’s structure, offers us the ideal blend of flexibility and organization."
            }
        ],
        "question": "Who is the Scrum Master in my report?",
        "answer": "The Scrum Master for the current project is Dr. Feryel Beji. She facilitates the Scrum process and supports the team. (Page 20)"
    },
    
    # Example 2: Sports
    {
        "context": [
            {
                "page_number": 50,
                "sentence_chunk": "Chapter 5: The History of Soccer. Soccer, known as football outside the United States, has a rich history dating back to ancient civilizations. Modern soccer began to take shape in England in the 19th century."
            },
            {
                "page_number": 51,
                "sentence_chunk": "The first official soccer match took place in 1863 in London, which also saw the establishment of The Football Association. This marked the beginning of organized soccer with standardized rules."
            }
        ],
        "question": "When and where did the first official soccer match take place?",
        "answer": "The first official soccer match took place in 1863 in London. (Page 51)"
    },
    
    # Example 3: Technology
    {
        "context": [
            {
                "page_number": 30,
                "sentence_chunk": "Chapter 3: The Evolution of Artificial Intelligence. Artificial intelligence (AI) has seen significant advancements since its inception. Early AI research in the 1950s focused on problem-solving and symbolic methods."
            },
            {
                "page_number": 31,
                "sentence_chunk": "In recent years, AI has expanded to include machine learning, where systems improve their performance through experience. Deep learning, a subset of machine learning, uses neural networks with many layers to analyze complex data patterns."
            }
        ],
        "question": "What are the recent advancements in AI?",
        "answer": "Recent advancements in AI include the development of machine learning and deep learning, which use neural networks to analyze complex data patterns. (Page 31)"
    },
    
    # Example 4: Literature
    {
        "context": [
            {
                "page_number": 10,
                "sentence_chunk": "Chapter 1: The Renaissance in Literature. The Renaissance period, spanning from the 14th to the 17th century, marked a revival of interest in the classical art, literature, and learning of ancient Greece and Rome."
            },
            {
                "page_number": 11,
                "sentence_chunk": "Key figures in Renaissance literature include Dante Alighieri, Geoffrey Chaucer, and William Shakespeare. Their works often explored themes of humanism, individualism, and the complexities of the human condition."
            }
        ],
        "question": "Who were some key figures in Renaissance literature?",
        "answer": "Key figures in Renaissance literature include Dante Alighieri, Geoffrey Chaucer, and William Shakespeare. (Page 11)"
    },
    
    # Example 5: Philosophy
    {
        "context": [
            {
                "page_number": 60,
                "sentence_chunk": "Chapter 6: Existentialism. Existentialism is a philosophical movement that emerged in the 20th century, emphasizing individual freedom, choice, and existence. It is often associated with philosophers such as Jean-Paul Sartre, Friedrich Nietzsche, and Søren Kierkegaard."
            },
            {
                "page_number": 61,
                "sentence_chunk": "Central themes in existentialism include the absurdity of life, the inevitability of death, and the necessity of making meaningful choices in an indifferent universe."
            }
        ],
        "question": "What are the central themes in existentialism?",
        "answer": "Central themes in existentialism include the absurdity of life, the inevitability of death, and the necessity of making meaningful choices in an indifferent universe. (Page 61)"
    },
    
    # Example 6: History
    {
        "context": [
            {
                "page_number": 75,
                "sentence_chunk": "Chapter 7: The Industrial Revolution. The Industrial Revolution, which began in the late 18th century, was a period of great technological innovation and economic change. It started in Britain and soon spread to other parts of the world."
            },
            {
                "page_number": 76,
                "sentence_chunk": "Key inventions of the Industrial Revolution include the steam engine, the spinning jenny, and the power loom. These innovations greatly increased production capabilities and efficiency."
            }
        ],
        "question": "What were some key inventions of the Industrial Revolution?",
        "answer": "Key inventions of the Industrial Revolution include the steam engine, the spinning jenny, and the power loom. (Page 76)"
    },
    
    # Example 7: Science
    {
        "context": [
            {
                "page_number": 40,
                "sentence_chunk": "Chapter 4: The Theory of Relativity. Albert Einstein's theory of relativity revolutionized our understanding of space, time, and gravity. The theory consists of two parts: special relativity and general relativity."
            },
            {
                "page_number": 41,
                "sentence_chunk": "Special relativity, introduced in 1905, deals with objects moving at constant speeds, particularly at speeds close to the speed of light. General relativity, introduced in 1915, provides a new description of gravity as the curvature of spacetime."
            }
        ],
        "question": "What are the two parts of Einstein's theory of relativity?",
        "answer": "Einstein's theory of relativity consists of two parts: special relativity, which deals with objects moving at constant speeds, and general relativity, which describes gravity as the curvature of spacetime. (Page 41)"
    },
    #Example8 Sports:
    {
        "Context": [],
        "Question": "What is the history of soccer?",
        "Answer": "Sorry, your question is irrelevant to the book's content. Please adjust your question to include specific details related to the book."
        },
    # Example 9: Art
    {
        "context": [
            {
                "page_number": 90,
                "sentence_chunk": "Chapter 9: The Impressionist Movement. The Impressionist movement, which began in the late 19th century, sought to capture the effects of light and color in everyday scenes. It marked a departure from traditional artistic techniques and subjects."
            },
            {
                "page_number": 91,
                "sentence_chunk": "Notable Impressionist artists include Claude Monet, Pierre-Auguste Renoir, and Edgar Degas. Their works often featured vibrant colors and loose brushwork to convey the impression of a moment in time."
            }
        ],
        "question": "Who are some notable Impressionist artists?",
        "answer": "Notable Impressionist artists include Claude Monet, Pierre-Auguste Renoir, and Edgar Degas. (Page 91)"
    },
   # Example 10:History-context irrelevant to question
    {
        
        "context": [{
            "page_number": 10,
            "sentence_chunk": "Chapter 1: Ancient Civilizations. This chapter explores the rise and fall of ancient civilizations across the globe, including their cultural achievements and societal structures.",
        }],
        "question": "What is the impact of the Industrial Revolution on ancient civilizations?",
        "answer": "Please adjust your question to focus on ancient civilizations or related topics covered in the book. (Chapter 1: Ancient Civilizations)"
            }
    ]
    

    prompt = f"""You are an assistant helping users to explore PDFs easily. I will provide you with context items, and you need to give clear and concise responses, including the page number where the related passages can be found.
    Context: {context_items}

    For each question, provide an easy-to-understand answer and specify the page number where the relevant information is located and make it more rich if its possible and you have knowledge about it.

    Examples:{examples}

    Now, respond to the following question:

    Question: {query}
    Answer:
    """
    
    gemini_response = st.session_state.chat_session.send_message(prompt)
    

    return gemini_response
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
def delete_pickle_files():
    if os.path.exists('embeddings.pkl'):
        os.remove('embeddings.pkl')
    if os.path.exists('pages_and_chunks.pkl'):
        os.remove('pages_and_chunks.pkl')
def main():
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    if st.session_state.pdf_uploaded and not pdf:
        delete_pickle_files()
        st.session_state.pdf_uploaded = False
    if os.path.exists('embeddings.pkl') and os.path.exists('pages_and_chunks.pkl'):
        with open('embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        with open('pages_and_chunks.pkl', 'rb') as f:
            pages_and_chunks = pickle.load(f)
    else:
        embeddings = None
        pages_and_chunks = None
    st.header("Chat with PDF 💬")
    
    MAX_UPLOAD_SIZE_MB = 30
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    pdf = st.file_uploader(f"Upload your PDF", type='pdf')
    query = st.text_input("Ask questions about your PDF file:")
    btn=st.button("Ask")
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    if pdf:
        if pdf.size > MAX_UPLOAD_SIZE_BYTES:
            st.error(f"File size is too large! Please upload a file smaller than {MAX_UPLOAD_SIZE_MB} MB.")
            return
        st.session_state.pdf_uploaded = True
        if embeddings is None  and pages_and_chunks is None: 
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
                
                embeddings =embedding_model.encode(text_chunks, batch_size=64, convert_to_tensor=True)  
                with open('embeddings.pkl', 'wb') as f:
                    pickle.dump(embeddings, f)

                with open('pages_and_chunks.pkl', 'wb') as f:
                    pickle.dump(pages_and_chunks, f)
        if btn:
            if query:
                if embeddings is None or pages_and_chunks is None:
                    st.error("Please upload and process a PDF first.")
                else:
                    with st.spinner('Generating response...'):
                        rep=ask(query,embedding_model,embeddings,pages_and_chunks)
        st.write("\n\n")
        for message in reversed(st.session_state.chat_session.history):
            if message.role=="user":
                match = re.search(r"Question:\s*(.*)\s*Answer:", message.parts[0].text)
                question = match.group(1)
                st.markdown("Question:"+question)
            else:
                with st.chat_message(translate_role_for_streamlit(message.role)):
                    st.markdown(message.parts[0].text)

if __name__ == "__main__":
    main()
