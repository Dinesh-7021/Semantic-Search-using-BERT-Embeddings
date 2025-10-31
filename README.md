#  Semantic Search Using BERT Embeddings
*A Modern Alternative to TF-IDF for Information Retrieval*

---

##  Overview

In the digital era, **Information Retrieval (IR)** plays a critical role in helping users access relevant information quickly and accurately.  
Traditional IR models like **TF-IDF** and **Boolean Retrieval** depend on exact keyword matching — often failing to capture **semantic meaning**.

This project implements a **Semantic Search System** using **BERT (Bidirectional Encoder Representations from Transformers)** to understand **context and intent** behind user queries.  
The model retrieves documents based on **meaning**, not just literal word matches.

---

##  Problem Statement

Traditional search systems rely on **keyword-based methods** such as TF-IDF and Boolean retrieval.  
These methods focus on literal word matching, ignoring **context** and **semantic similarity**.

For example:

> **Query:** “How to start a car engine?”  
> **Document:** “Steps to ignite a vehicle’s motor.”  

A TF-IDF model fails to recognize these as related.  
Hence, we need a **semantic search model** that can understand relationships and meaning, not just word frequency.

---

##  Objectives

- To compare **TF-IDF** and **BERT-based** retrieval mechanisms.  
- To build a **semantic search engine** using BERT sentence embeddings.  
- To visualize and analyze the difference in similarity scores.  
- To improve **relevance** and **accuracy** of retrieved results.

---

##  Features

- End-to-end pipeline for text preprocessing and document embedding.
- Implementation of **TF-IDF** and **BERT Sentence Embeddings**.
- **Cosine similarity** based document ranking.
- Clear **visual comparison** using bar and line graphs.
- Easy-to-extend and interpret code in Jupyter Notebook.

---

##  Methodology

###  Step 1: Text Preprocessing
- Tokenization  
- Stopword removal  
- Lowercasing and cleaning

###  Step 2: TF-IDF Approach
- Vocabulary extraction  
- Compute Term Frequency (TF)  
- Compute Inverse Document Frequency (IDF)  
- Multiply TF × IDF to form vectors  
- Compute cosine similarity between query and documents

###  Step 3: BERT-based Semantic Search
- Use **pre-trained BERT** to generate **contextual embeddings**
- Represent each document and query using the **[CLS] token embedding**
- Compute **cosine similarity** between query and document vectors
- Rank documents based on similarity scores

---

##  Algorithm: Semantic Representation using BERT

python
1. Start
2. Input sentence(s) → S = ["sentence A", "sentence B"]
3. Tokenization:
     tokens = WordPieceTokenizer(S)
     Add [CLS] at start and [SEP] between sentences
4. Convert tokens → token_ids
     Create attention_mask for non-padding tokens
5. Embedding Layer:
     input_embed = token_embed + pos_embed + seg_embed
6. Transformer Encoder:
     For each layer L in encoder layers:
         attention_output = MultiHeadSelfAttention(input_embed)
         output = FeedForward(attention_output)
         input_embed = LayerNorm(output)
7. Fine-tuning:
     Use [CLS] embedding as sentence representation
     Compute similarity = cosine(query_vec, doc_vec)
8. Rank documents by descending similarity scores
9. End


---

##  Results and Visualization

### **TF-IDF vs BERT Similarity Scores**

| Document             | TF-IDF Score | BERT Score |
| -------------------- | ------------ | ---------- |
| D2 – AI & ML         | 0.45         | 0.89       |
| D4 – Deep Learning   | 0.55         | 0.92       |
| D6 – Neural Networks | 0.36         | 0.87       |

* **BERT** captures contextual similarity and ranks relevant documents higher.
* **TF-IDF** fails to relate semantically similar sentences with different words.

### **Visual Comparison**

*  *Bar Graph:* BERT gives higher similarity for semantically related texts.
*  *Line Graph:* BERT’s curve is smoother — showing semantic consistency.

*(Refer to `output_graphs/` folder or notebook visualizations.)*

---

##  How to Run

1. Clone the repository:

   bash
   git clone https://github.com/<your-username>/Semantic-Search-BERT.git
   cd Semantic-Search-BERT
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   

3. Open the notebook:

   bash
   jupyter notebook Semantic_Search_BERT.ipynb
   

4. Run all cells to compare **TF-IDF** vs **BERT** search performance.

---

##  Complexity

| Model  | Time Complexity | Notes                                                          |
| ------ | --------------- | -------------------------------------------------------------- |
| TF-IDF | O(N × V)        | Linear with respect to number of documents and vocabulary size |
| BERT   | O(n² × d)       | Quadratic due to self-attention (n = tokens, d = dimension)    |

Despite higher computational cost, **BERT provides context-aware accuracy** far beyond traditional methods.

---

##  Technologies Used

* Python 
* Jupyter Notebook
* Scikit-learn
* Transformers (Hugging Face)
* NumPy & Pandas
* Matplotlib

---

##  Folder Structure


Semantic-Search-BERT/
│
├── Semantic_Search_BERT.ipynb      # Jupyter notebook
├── presentation.pptx               # Project presentation
├── requirements.txt                # Dependencies
├── output_graphs/                  # Visualization outputs
├── README.md                       # Project documentation
└── data/                           # Optional dataset folder

---

##  References

* Jacob Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.”
* Scikit-learn Documentation – TF-IDF Vectorizer
* HuggingFace Transformers Library

---

##  Author

**Dinesh Kumar Reddy Tatigutla**
📧 [dineshreddy7021@gmail.com]
💬 “Building smarter systems that understand meaning — not just words.”

---

## 🏁 Conclusion

This project demonstrates how **BERT embeddings** bridge the gap between **literal** and **contextual** retrieval.
By integrating deep contextual understanding, we move toward **AI-driven intelligent search systems** that deliver more **accurate** and **relevant** results.

---

