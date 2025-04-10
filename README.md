# Credit Card Fraudulent Transaction Detection Model

This project uses machine learning to detect fraudulent credit card transactions. It includes a Jupyter notebook for data exploration and model training, and a `Streamlit` app to serve predictions via a web interface.

## 📂 Project Structure

```
Credit-Card-Fraudulant-Transaction-Detection-Model-main/
├── app.py                                 # Streamlit app for prediction
├── creditcard.csv                         # Dataset of transactions
├── model.pkl                              # Trained ML model (Random Forest)
├── Credit Card Fraud Detection using Machine Learning.ipynb  # Model training notebook
└── Credit Card Detection Machine Learning Project with Report.docx  # Project documentation
```

## 📊 Dataset

The dataset contains transaction data and a binary class label indicating whether a transaction is fraudulent (`1`) or not (`0`).

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Credit-Card-Fraudulant-Transaction-Detection-Model-main.git
   cd Credit-Card-Fraudulant-Transaction-Detection-Model-main
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

4. **Explore the Notebook**
   Open the `.ipynb` file in Jupyter to view model training and evaluation.

## 🧠 Model

- **Technique**: Random Forest Classifier
- **Preprocessing**: Standard Scaler, SMOTE for class imbalance
- **Evaluation**: Accuracy, Precision, Recall, F1-score

## 📦 Requirements

See `requirements.txt` for all dependencies.

## ✍️ Author

Your Name  
*Feel free to connect and contribute!*
