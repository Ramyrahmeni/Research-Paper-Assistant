import streamlit as st
import PyPDF2
import pandas as pd
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer,CrossEncoder
import os
from stqdm import stqdm  
import torch
from dotenv import load_dotenv
import google.generativeai as gen_ai

import pickle
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-1.5-pro')
def re_rank(query, contexts):
    cross_inp = [[query, ctx['text']] for ctx in contexts]
    scores = cross_encoder.predict(cross_inp)
    sorted_contexts = [x for _, x in sorted(zip(scores, contexts), key=lambda pair: pair[0], reverse=True)]
    return sorted_contexts
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

def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """Splits the input_list into sublists of size slice_size (or as close as possible)."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def elimination_chunks(df: pd.DataFrame, min_token_length) -> list[dict]:
    pages_and_chunks_over_min_token_len = df[df["page_token_count"] > min_token_length].to_dict(orient="records")

    return pages_and_chunks_over_min_token_len

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                embedding_model: SentenceTransformer,
                                n_resources_to_return: int = 4):
    """
    Embeds a query with the model and returns top K scores and indices from embeddings.
    """
    # Embed the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Get dot product scores on embeddings
    dot_scores = embedding_model.similarity(query_embedding, embeddings)[0]
    
    # Get the number of scores above a threshold (e.g., 0.5)
    valid_scores = dot_scores > 0.2
    cnt = valid_scores.sum().item()
    
    # Get top K scores and their indices
    top_k = min(cnt, n_resources_to_return)
    scores, indices = torch.topk(dot_scores, k=top_k)
    
    return scores, indices

def ask(query, tables,embedding_model, embeddings, pages_and_chunks):
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
    #context_items = re_rank(query, context_items)
    print(context_items)
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
            {
        
    "context": {
        "page_number": 160,
        "sentence_chunk": "Table 15: Laptop Specifications. This table provides a comparison of two laptops based on their operating system, RAM, and processor."
    }
,
        "question": "How do Laptop 1 and Laptop 2 compare in terms of their specifications?",
        "answer": """The specifications of Laptop 1 and Laptop 2 are compared as follows:
Operating System: Laptop 1 runs on Windows 11 Professional, while Laptop 2 uses Windows 10 Professional.
RAM: Laptop 1 is equipped with 24 GB of RAM, which is double the 12 GB RAM available in Laptop 2, making it more suitable for multitasking and intensive applications.
Processor: Laptop 1 features an 11th Gen Intel(R) Core(TM) i7-1165G7 processor, offering higher performance compared to Laptop 2's Intel(R) Core(TM) i5-8265U CPU. The i7 processor in Laptop 1 provides better speed and efficiency, which is beneficial for demanding computational tasks. (Table 15, Page 160)"""
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
            },
            {
        
    "context":{
        "page_number": 140,
        "sentence_chunk": "Table 14: Performance Metrics of Various Models. This table compares the accuracy, precision, recall, and F1-score of different machine learning models used for email classification."
    }
,
        "question": "How do the machine learning models compare in terms of performance metrics?",
        "answer": """The performance metrics for various machine learning models used for email classification are as follows:
Random Forest Classifier: Exhibits strong performance with an accuracy of 93%, precision of 94%, recall of 94%, and F1-score of 94%.
Support Vector Machine: Shows lower performance with an accuracy of 50%, precision of 24%, recall of 49%, and F1-score of 33%.
BERT Model: Utilizes BERT for accurate email categorization, achieving an accuracy of 89%, precision of 88%, recall of 92%, and F1-score of 90%.
Complement Naive Bayes: Optimized for imbalanced datasets, it has an accuracy of 88%, precision of 85%, recall of 98%, and F1-score of 91%.
Bernoulli Naive Bayes: Assumes features are binary and performs exceptionally well with an accuracy of 96%, precision of 96%, recall of 97%, and F1-score of 96.7%. (Table 14, Page 140)"""
            }
    ]
    

    prompt = f"""
    You're assisting users in navigating PDFs effectively. Provide clear and concise responses, including page numbers for relevant passages. Utilize data from tables within the PDF for detailed analysis, if available. If there's no context but relevant tables exist, use them for your response.

    Context: {context_items}
    Examples: {examples}

    Question: {query}
    Answer:
    """

    print(prompt)
    
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
    if os.path.exists('tables.pkl'):
        os.remove('tables.pkl')
def main():
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    
    if os.path.exists('embeddings.pkl') and os.path.exists('pages_and_chunks.pkl') and os.path.exists('tables.pkl'):
        with open('embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        with open('pages_and_chunks.pkl', 'rb') as f:
            pages_and_chunks = pickle.load(f)
        with open('tables.pkl', 'rb') as f:
            tables = pickle.load(f)
    else:
        embeddings = None
        pages_and_chunks = None
        tables=None
    st.header("Chat with PDF 💬")
    
    MAX_UPLOAD_SIZE_MB = 5
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    pdf = st.file_uploader(f"Upload your PDF", type='pdf')
    if st.session_state.pdf_uploaded and not pdf:
        delete_pickle_files()
        st.session_state.chat_session = model.start_chat(history=[])          
        st.session_state.pdf_uploaded = False
    query = st.text_input("Ask questions about your PDF file:")
    btn=st.button("Ask")
    embedding_model = SentenceTransformer("all-mpnet-base-v2",device='cpu')
    if pdf:
        if pdf.size > MAX_UPLOAD_SIZE_BYTES:
            st.error(f"File size is too large! Please upload a file smaller than {MAX_UPLOAD_SIZE_MB} MB.")
            return
        st.session_state.pdf_uploaded = True
        if embeddings is None  and pages_and_chunks is None: 
            with st.spinner('Processing PDF...'):
                pages_and_texts = open_and_read_pdf(pdf)

                # Extract tables from the PDF
                nlp = English()
                nlp.add_pipe("sentencizer")
                text_chunks=[]
                for item in stqdm(pages_and_texts, desc="Tokenizing pages"):
                    item["sentences"] = list(nlp(item["text"]).sents)
                    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
                    item["page_sentence_count_spacy"] = len(item["sentences"])
                    text_chunks.append(item["sentences"])
                
                pages_and_chunks=pages_and_texts
                df = pd.DataFrame(pages_and_chunks)
                pages_and_chunks = elimination_chunks(df, 30)
                #slice_size = round((340 * sent) / token)
                
                #for item in stqdm(pages_and_texts, desc="Splitting sentences into chunks"):
                 
                 #   item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=slice_size)
                  
                  #  item["num_chunks"] = len(item["sentence_chunks"])
                
                #pages_and_chunks = pages_chunks(pages_and_texts)
                #df = pd.DataFrame(pages_and_chunks)
                #pages_and_chunks = elimination_chunks(df, 30)
                #text_chunks = [item["sentence_chunks"] for item in pages_and_chunks]

                embeddings =embedding_model.encode(text_chunks, batch_size=512, convert_to_tensor=True)  
                with open('embeddings.pkl', 'wb') as f:
                    pickle.dump(embeddings, f)

                with open('pages_and_chunks.pkl', 'wb') as f:
                    pickle.dump(pages_and_chunks, f)
                with open('tables.pkl', 'wb') as f:
                    pickle.dump(tables, f)
        if btn:
            if query:
                if embeddings is None or pages_and_chunks is None:
                    st.error("Please upload and process a PDF first.")
                else:
                    with st.spinner('Generating response...'):
                        rep=ask(query,tables, embedding_model,embeddings,pages_and_chunks)
        st.write("\n\n")
        for message in reversed(st.session_state.chat_session.history):
            if message.role=="user":
                match = re.search(r"Question:\s*(.*)\s* ", message.parts[0].text)
                question = match.group(0).strip()
                with st.chat_message(translate_role_for_streamlit(message.role)):
                    st.markdown(question)
            else:
                with st.chat_message(translate_role_for_streamlit(message.role)):
                    st.markdown(message.parts[0].text)



if __name__ == "__main__":
    main()
