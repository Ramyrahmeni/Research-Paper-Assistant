# 🤗💬 LLM Chat App

## Overview
The LLM Chat App is a Streamlit-based application that enables users to upload PDF files and interactively ask questions about the content of the PDFs. The application leverages advanced language models, including Google's Generative AI model (gemini-1.5-pro) and the SentenceTransformer model, to provide accurate and context-aware responses to user queries.

## Features
- **PDF Upload**: Users can upload PDF files for processing.
- **Content Extraction**: Extracts text content from PDF pages and processes it for analysis.
- **Interactive Q&A**: Users can ask questions about the content of the uploaded PDF and receive detailed answers.
- **Contextual Understanding**: Utilizes sentence embeddings to understand and retrieve relevant content.
- **Streamlit UI**: Provides a user-friendly interface for seamless interaction.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/llm-chat-app.git
    cd llm-chat-app
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the environment variables:**
    - Create a `.env` file in the root directory of the project.
    - Add your Google API key to the `.env` file:
        ```
        API_KEY=your_google_api_key_here
        ```

## Usage
1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Upload a PDF:**
    - Click on the "Browse files" button in the sidebar to upload your PDF file.
    - Ensure the PDF file size is less than 5 MB.

3. **Ask Questions:**
    - Enter your query in the text input box and click the "Ask" button.
    - The app will process the PDF and provide an answer based on the content of the PDF.

## Project Structure
- `app.py`: Main application script.
- `requirements.txt`: List of dependencies required for the project.
- `.env`: Environment file for storing API keys and other configurations.
- `README.md`: Project documentation file.

## Key Components
### PDF Processing
- **Text Extraction**: Uses PyPDF2 to extract text from each page of the PDF.
- **Sentence Tokenization**: Utilizes SpaCy's `sentencizer` to split the extracted text into sentences.
- **Chunking**: Splits sentences into manageable chunks for embedding and analysis.

### Vector Search Process
The vector search process involves the following steps:

1. **Embedding Text Chunks**:
    - The extracted text from the PDF is divided into smaller, manageable chunks based on sentences.
    - Each text chunk is converted into a numerical representation (embedding) using the SentenceTransformer model (`all-mpnet-base-v2`).
    - Sentence embeddings capture the semantic meaning of the text, allowing for effective comparison and retrieval.

2. **Storing Embeddings**:
    - The embeddings of all text chunks are stored in a list.
    - These embeddings are saved to a file (`embeddings.pkl`) for quick access and reusability.

3. **Query Embedding**:
    - When a user inputs a query, the query is also converted into an embedding using the same SentenceTransformer model.
    - This ensures that both the query and the text chunks are in the same vector space, making similarity comparison possible.

4. **Similarity Calculation**:
    - The similarity between the query embedding and each text chunk embedding is calculated using the dot product.
    - A higher dot product score indicates greater similarity between the query and the text chunk.

5. **Retrieving Relevant Chunks**:
    - The top N text chunks with the highest similarity scores are retrieved.
    - These relevant text chunks are used to provide context for generating the final response to the user's query.

6. **Generating Response**:
    - The relevant text chunks are provided as context to the Generative AI model (gemini-1.5-pro).
    - The model generates a response based on the context and the query, ensuring the answer is accurate and relevant to the content of the PDF.

#### Code Snippets for Vector Search

**Embedding Text Chunks:**
```python
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
embeddings = embedding_model.encode(text_chunks, batch_size=64, convert_to_tensor=True)

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
```

**Query Embedding and Similarity Calculation:**
```python
def retrieve_relevant_resources(query, embeddings, embedding_model, n_resources_to_return=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    valid_scores = dot_scores > 0.1
    cnt = valid_scores.sum().item()
    top_k = min(cnt, n_resources_to_return)
    scores, indices = torch.topk(dot_scores, k=top_k)
    return scores, indices
```

### Generative AI
- **Google Generative AI**: Configures and uses Google's gemini-1.5-pro model to generate responses based on the context and query.
- **Chat History Management**: Maintains chat history using Streamlit's session state.

### Streamlit UI
- **File Upload**: Provides a file uploader widget for PDF files.
- **Query Input**: Allows users to input their queries interactively.
- **Response Display**: Shows the generated response and relevant context from the PDF.

## Example Queries
- "Who is the Scrum Master in my report?"
- "When and where did the first official soccer match take place?"
- "What are the recent advancements in AI?"

## Contributing
We welcome contributions to enhance the functionality and features of this project. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypdf2.readthedocs.io/en/latest/)
- [SpaCy](https://spacy.io/)
- [SentenceTransformer](https://www.sbert.net/)
- [Google Generative AI](https://cloud.google.com/generative-ai)

---

This README file provides comprehensive information about the project, including setup instructions, usage guidelines, and detailed explanations of the vector search process. Adjust the URLs and other details as needed for your specific repository and environment.
