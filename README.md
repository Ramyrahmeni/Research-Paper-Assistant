# Research Paper Assistant

Welcome to the Research Paper Assistant! This app allows you to upload a PDF of a research paper, process it into chunks, and ask questions about its content. The app uses a Language Model (LLM) to understand and respond to your queries based on the content of the PDF. This README will guide you through the installation, usage, and understanding of the code.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Uploading PDF](#uploading-pdf)
  - [Chunking the Text](#chunking-the-text)
  - [Vector Search](#vector-search)
  - [Feeding Data to LLM](#feeding-data-to-llm)
- [Detailed Pipeline](#detailed-pipeline)
- [Features](#features)
- [Challenges](#challenges)
- [Justification for Choosing Gemini 1.5 Pro](#justification-for-choosing-gemini-1.5-pro)

## Installation

To run this app locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ramyrahmeni/Research-Paper-Assistant.git
   cd Research-Paper-Assistant
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables:**
   Create a `.env` file in the root of your project and add your Google API key:
   ```
   API_KEY=your_google_api_key_here
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload a PDF:**
   - Go to the sidebar, click on "Upload your Research Paper" and select a PDF file.

2. **Ask Questions:**
   - Type your question in the text input field and click the "Ask" button.

3. **View Responses:**
   - The app will display the question and the LLM-generated response based on the content of the PDF.

## Code Explanation

### Uploading PDF

The `open_and_read_pdf` function handles the uploading and reading of the PDF file. It extracts text content from each page and collects statistics such as character count, word count, and sentence count.

### Chunking the Text

Text chunking is done to process the PDF content into manageable pieces. This is achieved using SpaCy for sentence segmentation and further processing to ensure the chunks are suitable for embedding.

### Vector Search

The vector search functionality is implemented using the `SentenceTransformer` model from the `sentence_transformers` library. This model is used to encode the text chunks and the user queries into vectors, enabling similarity searches to find the most relevant chunks for a given query.

### Feeding Data to LLM

The `ask` function handles the integration with the LLM. It retrieves relevant chunks based on the query, constructs a prompt, and sends it to the Generative Model (`gemini-1.5-pro`). The response is then displayed in the Streamlit app.

## Detailed Pipeline

1. **Uploading and Reading PDF:**
   - The PDF is uploaded using Streamlit's file uploader.
   - `PyPDF2` reads the PDF and extracts text from each page.
   - Text is formatted using the `text_formatter` function.

2. **Text Chunking:**
   - The text is tokenized into sentences using SpaCy's `sentencizer` pipeline component. SpaCy is a powerful NLP library that allows for efficient text processing.
   - Chunks are created based on a minimum token length (set to 30 tokens in this case) to ensure meaningful pieces. Token length is approximated by dividing the character count by 4.

3. **Embedding:**
   - Text chunks are encoded into vectors using the `all-mpnet-base-v2` model from the `sentence_transformers` library. This model leverages the transformer architecture, specifically a variant of BERT (Bidirectional Encoder Representations from Transformers) optimized for sentence embeddings.
   - Embeddings are computed on the CPU to accommodate environments without GPUs. These embeddings are saved using Python's `pickle` module for quick retrieval in future sessions.

4. **Vector Search and Retrieval:**
   - User queries are also encoded into vectors using the same `SentenceTransformer` model.
   - Similarity search is performed using dot product similarity to find the most relevant text chunks. The `torch.topk` function is used to retrieve the top K most relevant chunks based on their similarity scores.

5. **Query Response:**
   - A prompt is created with examples and context items derived from the most relevant text chunks. This prompt follows a specific format to ensure the LLM can generate coherent and accurate responses.
   - The prompt is sent to the LLM (`gemini-1.5-pro`), which is configured using the Google Generative AI API.
   - The response from the LLM is displayed to the user in the Streamlit app, utilizing the `st.session_state` to manage the chat history.

## Features

- **Technical Proficiency:** The model excels in answering technical questions, leveraging its ability to understand and process complex technical language and concepts effectively.
- **Table-Based Responses:** The model provides clear and accurate table-based answers, making it easy to understand data presented in tabular formats.
- **Precise Table and Image Analysis:** The model can analyze tables and images precisely, extracting relevant information and presenting it in a comprehensible manner.
- **Comprehensive Analysis:** The model offers thorough analysis capabilities, ensuring detailed and insightful responses based on the content of the PDF.

## Challenges

- **General Question Handling:** The model sometimes struggles with general questions that are not directly related to the specific content of the PDF. Improving the model's ability to handle these questions is an ongoing area of research.
- **Context Relevance:** Ensuring the relevance of the retrieved chunks to the user's query can be challenging, especially with ambiguous or broadly phrased questions. Enhancing the model's contextual understanding is essential for better performance.
- **Scalability:** Processing large PDFs with numerous pages and complex structures can be resource-intensive. Optimizing the pipeline for scalability and efficiency remains a challenge.

## Justification for Choosing Gemini 1.5 Pro

After evaluating both Llama 3 and Gemini 1.5 Pro, we chose Gemini 1.5 Pro due to its superior performance in several critical areas:

- **Accuracy**: Gemini 1.5 Pro consistently provided accurate answers in technical domains, particularly in probability and statistical analysis.
- **Validation**: The model effectively validated its answers using various methods, ensuring reliability.
- **Complex Problem Solving**: Gemini 1.5 Pro demonstrated robust handling of complex reasoning tasks, making it well-suited for our application's needs.
- **Consistency**: In tests, Gemini 1.5 Pro showed consistent performance, whereas Llama 3 exhibited variability, especially in handling more challenging problems.

By leveraging Gemini 1.5 Pro, we ensure that our application delivers precise, reliable, and insightful responses to users, especially for technically demanding queries.