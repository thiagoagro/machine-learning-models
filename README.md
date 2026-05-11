# Automated Machine Learning Platform

Welcome to the **Automated Machine Learning SaaS Platform**, an end-to-end ecosystem designed to democratize Machine Learning. This project features a robust Python package (`ml_package`) for data processing and model training, accompanied by a sleek, interactive web interface powered by **Streamlit**. 

Whether you are a researcher, a business analyst with no coding experience, or a data scientist looking to speed up your pipeline, this platform has you covered.

---

## 🌟 Key Features

1. **Advanced Exploratory Data Analysis (EDA)**
   - Automatically detects numerical and categorical columns.
   - Interactive scatter plots, distribution histograms, and heatmaps for correlation matrix using **Plotly**.
   - Missing data detection.

2. **Intelligent Feature Selection**
   - Automatically identifies the most impactful variables for your target using `ExtraTrees` algorithms.
   - Suggests the subset of features that explain 95% of your dataset's variance/information.

3. **Automated Model Training (AutoML)**
   - Supports both **Regression** and **Classification** tasks.
   - Dynamic selection from a vast dictionary of algorithms (Random Forest, XGBoost, SVM, Gradient Boosting, LightGBM, Neural Networks, etc.).
   - Built-in **Hyperparameter Tuning** via `RandomizedSearchCV`.
   - Scalability options (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).

4. **Visual Performance & Reporting**
   - Compares Training and Test datasets on the same Interactive Plotly graph to easily spot overfitting or underfitting.
   - Generates interactive Confusion Matrices for classification tasks.
   - 1-Click Export: Downloads all performance metrics, model rankings, and predictions into a multi-sheet **Excel (.xlsx)** file.

---

## 🏗️ Architecture

The project is strictly decoupled into two main layers:

```text
ML_Project/
├── ml_package/                 # Core Backend (The "Brain")
│   ├── __init__.py             
│   ├── data_processor.py       # Data loading and EDA generation
│   ├── features.py             # Feature scaling and Importance evaluation
│   ├── trainer.py              # Training loop, CV, and GridSearch logic
│   ├── evaluation.py           # Statistical metrics (R2, RMSE, Accuracy, F1...)
│   ├── plot_utils.py           # Plotly-based graphing utilities
│   └── exporter.py             # Multi-sheet Excel exporter
│
├── app/                        # Frontend Application
│   └── main.py                 # Streamlit entry point
│
└── requirements.txt            # Project dependencies
```

---

## 🚀 Installation

**Prerequisites:** Python 3.9 or higher.

1. Clone the repository or download the source code.
2. Open your terminal and navigate to the project directory:
   ```bash
   cd ML_package
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ How to Run the Web App (SaaS)

To start the interactive interface, run the following command in your terminal:

```bash
streamlit run app/main.py
```

### User Workflow in the App:
1. **Upload:** Upload your dataset (`.csv` or `.xlsx`) in the sidebar.
2. **EDA Tab:** Explore your data distribution and correlations.
3. **Feature Selection Tab:** Select your Target variable and let the AI compute the most important features.
4. **Setup Tab:** Confirm your inputs, choose the scaling method, select the Machine Learning models you want to train, and hit **Start Training**.
5. **Results Tab:** View the leaderboard of your models, analyze the interactive plots (Observed vs Predicted), and download the Excel Report.

---

## 💻 Using the Python Package Programmatically

If you are a developer, you can bypass the Streamlit app and use `ml_package` directly in your Python scripts or Jupyter Notebooks.

```python
import pandas as pd
from ml_package import data_processor, features, trainer

# 1. Load Data
df = pd.read_csv("my_dataset.csv")

# 2. Select Features and Target
input_cols = ['Feature_1', 'Feature_2', 'Feature_3']
target_col = 'Target_Variable'

# 3. Train Models
results_df, predictions_df = trainer.run_training_pipeline(
    df=df,
    features_cols=input_cols,
    target_col=target_col,
    task_type='Regressão',              # Or 'Classificação'
    scaler_option='StandardScaler',     
    use_gridsearch=True,                # Enable Hyperparameter Tuning
    selected_models=['Random Forest', 'XGBoost', 'SVM (RBF)']
)

# 4. View Results
print(results_df.head())
```

---

## 🤝 Contributing

Contributions are welcome! If you want to add a new model algorithm, simply instantiate it in the dictionaries inside `ml_package/models.py` — the Streamlit UI will automatically pick it up and display it as an option.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
