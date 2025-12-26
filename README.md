# 📰 Automated Fake News Detection System (98% Accuracy)

### 🚀 Project Overview
In the era of information overload, distinguishing between credible and fabricated news is a critical challenge. This project implements a high-accuracy machine learning pipeline to classify news articles as **Real** or **Fake**. By utilizing Natural Language Processing (NLP) and ensemble learning, the system achieves a robust **98.05% accuracy** on large-scale datasets.

### 🧠 Core Methodology & Logic
The system processes raw text through a multi-stage pipeline designed for high-speed classification:

1.  **Text Preprocessing:**
    * **Normalization:** Lowercasing, removing punctuation, and stripping "noise" (URLs, special characters) using `re` (Regular Expressions).
    * **Stopword Removal:** Eliminating common words that do not carry sentimental or factual weight to reduce feature dimensionality.
2.  **Feature Extraction:**
    * **TF-IDF Vectorization:** Utilized `TfidfVectorizer` to convert text into numerical vectors. This method prioritizes "unique" keywords that help distinguish fake news patterns from standard journalistic style.
3.  **Model Architecture:**
    * **Passive Aggressive Classifier:** Ideally suited for large-scale data streams; it remains passive for correct classifications but "aggressively" updates weights upon encountering errors.
    * **Random Forest Classifier:** An ensemble method used to validate results and prevent overfitting through multiple decision trees.

### 📊 Performance Metrics
- **Accuracy:** 98.05%
- **Dataset Size:** 40,000+ news articles (merged True and Fake datasets).
- **Inference Speed:** Optimized for real-time text analysis.

### 🛠 Tech Stack
- **Languages:** Python (Anaconda Environment)
- **ML Libraries:** Scikit-Learn, Pandas, NumPy
- **NLP Tools:** NLTK, Regular Expressions (re)
- **Environment:** Jupyter Notebook / Anaconda

### 📁 Repository Structure
- `Fake_News_Detection.ipynb`: Main research and model training notebook.
- `requirements.txt`: List of Python dependencies for local replication.
- `data/`: (Optional) Placeholder for the dataset link.

### 🚀 How to Run
1. Clone this repository: `git clone https://github.com/Ayushimishra02/Fake-News-Detection-ML.git`
2. Install dependencies: `pip install wordcloud`
3. Launch Jupyter Notebook and run `Fake_News_Detection.ipynb`.
