# HelpMateAI-Project

![HelpMateAI](https://github.com/ivineettiwari/HelpMateAI-Project/raw/main/assets/banner.png)  

## 🚀 Introduction  

**HelpMateAI** is an **AI-powered Retrieval-Augmented Generation (RAG) system** designed to **retrieve relevant information from insurance documents** and **generate accurate responses using LLMs like GPT-3.5**. It combines **vector search**, **cross-encoder re-ranking**, and **LLM-based response generation** to provide fact-based, context-aware answers to user queries.  

---

## 🏗️ Architecture Overview  

### **1. Document Processing & Vector Store**
- Extracts text from PDFs using **pdfplumber**.
- Splits text into **fixed-size chunks** for embedding.
- Converts chunks into vector embeddings using **SentenceTransformer**.
- Stores embeddings in a **vector database (ChromaDB)**.

### **2. Retrieval & Re-ranking**
- Searches relevant document chunks using **vector similarity search**.
- Reranks results using a **Cross-Encoder** model for improved accuracy.

### **3. Response Generation**
- Uses **GPT-3.5 Turbo** for generating responses based on retrieved documents.
- Ensures **citations** are provided with each response.

---

## 📌 Features  

✅ **Advanced Semantic Search** - Finds the most relevant text using **ChromaDB**  
✅ **AI-Powered Re-Ranking** - Uses a **Cross-Encoder** model to improve ranking accuracy  
✅ **Natural Language Querying** - Accepts **user-friendly queries** and provides **detailed answers**  
✅ **Retrieval-Augmented Generation (RAG)** - Uses GPT-3.5 to generate responses **based on real documents**  
✅ **Caching for Faster Retrievals** - Speeds up responses using a **query cache**  
✅ **Insurance Domain Specific** - Designed for **insurance-related questions**  

---

## 🛠️ Installation & Setup  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ivineettiwari/HelpMateAI-Project.git
cd HelpMateAI-Project
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```bash
python -m venv env
source env/bin/activate  # On Mac/Linux
env\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys**
Create a `.env` file in the root directory and add your **OpenAI API Key**:  
```ini
OPENAI_API_KEY=your-api-key-here
```

---

## ⚡ Usage  

### **Run the Application**
```bash
python main.py
```

### **Query Processing Example**
```python
query = "What is the life insurance coverage for disability?"
df = search(query)
df = apply_cross_encoder(query, df)
df = get_topn(3, df)
response = generate_response(query, df)
print("\n".join(response))
```

### **Expected Output**
```
Life insurance coverage for disability typically depends on specific policy terms. Please review the relevant sections in the policy document.

Citation:
- Policy Name: Member Life Insurance or Coverage During Disability
- Page Number: 42
```

---

## 🏗️ Project Structure  

```
📂 HelpMateAI-Project
│── 📂 data/              # Processed document embeddings & cache
│── 📂 models/            # Pre-trained models for embedding & ranking
│── 📂 utils/             # Helper functions for text processing
│── 📂 api/               # API endpoints for external integration
│── main.py               # Main script to process user queries
│── requirements.txt      # Dependencies list
│── README.md             # Project documentation
│── .env                  # API keys (ignored in Git)
```

---

## 🚀 API Integration  

You can integrate this project into your own applications using the provided API.

### **Run the API Server**
```bash
uvicorn api.server:app --reload
```

### **Example API Query**
```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query": "What is ADL disability proof?"}'
```

### **Expected API Response**
```json
{
  "query": "What is ADL disability proof?",
  "response": "ADL Disability Proof refers to medical documentation required to certify...",
  "citations": [
    {"policy_name": "Life Insurance Policy", "page_number": 42}
  ]
}
```

---

## 🤝 Contribution  

Want to contribute? Great! Follow these steps:  

1. **Fork** the repo and create a new branch.  
2. **Make your changes** and test them locally.  
3. **Submit a Pull Request** with a description of the updates.  

---
