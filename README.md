# Financial Reports Chatbot: Your AI Analyst for Public Company Financials

![Financial Reports Chatbot](metadata/streamlit_finance_chatbot.png)

## Overview

Welcome to the Financial Reports Chatbot, a powerful conversational AI designed to provide instant answers to your financial questions about major publicly traded companies. This chatbot leverages cutting-edge Retrieval-Augmented Generation (RAG) technology, processing financial reports (PDFs) to deliver accurate, data-driven responses, minimizing guesswork and maximizing reliability.

Imagine having a financial analyst at your fingertips, ready to answer questions about revenue, expenses, and key performance indicators based on official company reports. This chatbot makes it a reality.

## Key Features

* **Intelligent PDF Processing**: Automatically extracts text from financial reports in PDF format, building a robust knowledge base.
* **Efficient Vector Database**: Utilizes Chroma to create a vector database, enabling rapid and accurate similarity searches for relevant information.
* **Conversational RAG**: Combines a powerful retrieval system with generative AI (powered by OpenAI) to provide contextually accurate and insightful answers.
* **User-Friendly Interface**: Built with Streamlit, offering a seamless and intuitive experience for uploading documents and interacting with the chatbot.
* **Robust Error Handling & Logging**: Ensures stability and provides detailed logs for debugging and optimization.
* **Automated FAQ Generation**: Generates relevant and insightful FAQs based on typical user queries and financial data.

## Technology Stack

* **Python**: The core programming language for the chatbot.
* **Streamlit**: For creating the interactive web application.
* **langchain**: The framework for building and managing the conversational AI pipeline.
* **OpenAI**: Powers the embeddings and generative language models.
* **Chroma (langchain_community)**: For efficient vector storage and retrieval.

## Getting Started

### Prerequisites

* Python 3.8 or higher
* pip (Python package installer)

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/nightlessbaron/financial_reports_chatbot.git](https://github.com/nightlessbaron/financial_reports_chatbot.git)
    cd financial_reports_chatbot
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**

    * Create a `.env` file in the root directory of the project.
    * Add your API keys to the `.env` file:

    ```plaintext
    TOGETHER_API_KEY='your_together_api_key_here'
    LANGCHAIN_API_KEY='your_langchain_api_key_here'
    LANGCHAIN_TRACING_V2='your_langchain_tracing_v2_here'
    ```

### Running the Application

To launch the chatbot, execute the following command:

```bash
streamlit run app.py



Open your web browser and navigate to http://localhost:8501 to access the application.

How to Use
Upload Financial Reports: Drag and drop your company's financial reports (PDFs) into the designated upload area.
Process PDFs: Click the "Process PDFs" button to extract the text and create the vector database.
Ask Your Questions: Enter your financial inquiries in the text input box and press enter. The chatbot will provide answers based on the uploaded reports.
Frequently Asked Questions (FAQs)
How does the chatbot generate FAQs?
The chatbot intelligently generates FAQs by combining predefined company names, financial terms, and years. It then uses the RAG chain to produce answers, providing valuable insights into common financial queries.

Can I customize the FAQ generation?
Yes! You can customize the FAQ generation by modifying the lists of companies, financial terms, and years within the generate_faqs function in app.py. This allows you to tailor the FAQs to specific industries or financial metrics.

Examples of Generated FAQs:
Q: What was Amazon's net income in 2023?
A: Amazon's net income in 2023 was $X billion, representing a Y% change from the previous year.
Q: How did Google's operating expenses change in Q2 2024?
A: Google's operating expenses in Q2 2024 increased by Z%, driven by [reasons from the report].
Q: What were Meta's key revenue drivers in the last fiscal year?
A: Meta's primary revenue drivers in the last fiscal year included [drivers listed from the report].
These FAQs are dynamic, and automatically change based on the pdf reports that are uploaded.
