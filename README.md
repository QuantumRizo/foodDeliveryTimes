# Food Delivery Time Prediction 🚚⏱️

Predicting food delivery times using machine learning and deep learning  
**Author:** David | Data Scientist

---

## Overview

This project uses real-world food delivery data to predict delivery times based on features like distance, weather, traffic, and more.  
It demonstrates a full ML workflow: data cleaning, feature engineering, model building (TensorFlow), and visual evaluation.

---

## Project Structure

```
foodDelivery/
├── data/                  # Raw data files
├── notebooks/             # Jupyter notebooks (EDA, prototyping)
│   └── foodDelivery.ipynb
├── src/                   # Python scripts for modular workflow
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

---

## Workflow

1. **Data Preprocessing**  
   - Cleans and engineers features (distance, time, categorical encoding)
   - Handles missing values and outliers

2. **Modeling**  
   - Deep learning regression model (TensorFlow/Keras)
   - Combines categorical and numerical features

3. **Training & Evaluation**  
   - Model checkpointing for best performance
   - Visualizes actual vs predicted delivery times

---


## Quickstart

```bash
# Clone the repo
git clone https://github.com/QuantumRizo/foodDelivery.git
cd foodDelivery

# Install dependencies
pip install -r requirements.txt

# Run training and evaluation
python src/train.py
python src/evaluate.py
```

---

## Notebooks

- **foodDelivery.ipynb**  
  Step-by-step EDA, feature engineering, and model prototyping with rich visualizations.

---

## Results

- **Best Test MAE:** 4.7
![Actual vs Predicted Delivery Times](notebooks/actual_vs_predicted.png)

---

## Contact

- [GitHub](https://github.com/QuantumRizo)

---

*Made with ❤️ by a Data Scientist who loves visual storytelling!*
