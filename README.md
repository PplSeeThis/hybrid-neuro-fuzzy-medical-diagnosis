# Hybrid Neuro-Fuzzy System for Medical Diagnosis

This project explores the development of a hybrid system that combines a neural network with a fuzzy logic system for predicting heart disease. The system uses the Heart Disease UCI dataset and aims to leverage the predictive power of neural networks and the interpretability of fuzzy logic.

## 📜 Project Overview

The project involved two main components:
1.  **Neural Network (NN):** A standard feed-forward neural network was built using PyTorch to predict the probability of heart disease based on patient data. This model achieved an accuracy of **75%** on its own.
2.  **Fuzzy Logic System:** A fuzzy inference system was created using `scikit-fuzzy`. It used linguistic variables (e.g., "young," "middle," "senior" for age) and a set of rules to assess risk.

These two components were then integrated. The output of the neural network (disease probability) was fed as an additional input into the fuzzy logic system, creating a more comprehensive hybrid model.

## 🛠️ Technologies & Libraries

* **Language:** Python
* **Core Libraries:**
    * PyTorch (for the Neural Network)
    * Scikit-fuzzy (for the Fuzzy Logic System)
    * Scikit-learn (for data preprocessing)
    * Pandas & NumPy

## 📈 Results & Conclusion

This project successfully demonstrated the integration of two distinct AI techniques.
* **Neural Network Accuracy:** **75.00%**
* **Hybrid System Accuracy:** **40.76%**

While the standalone neural network achieved higher raw accuracy, the primary goal was to explore the hybrid architecture. The fuzzy logic component makes the final decision-making process more transparent and interpretable, which is highly valuable in the medical domain. The lower accuracy of the hybrid system suggests that the fuzzy rules could be further refined or expanded. This project serves as a strong proof-of-concept for combining "black-box" models like neural networks with more transparent, rule-based systems.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/hybrid-neuro-fuzzy-medical-diagnosis.git](https://github.com/your-username/hybrid-neuro-fuzzy-medical-diagnosis.git)
    cd hybrid-neuro-fuzzy-medical-diagnosis
    ```
2.  **Create and activate a virtual environment.**
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main script:**
    ```bash
    python src/main.py
    ```
