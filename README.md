# Linear_Regression_pipeline
This is for dataset agnostic pipeline which uses imputer to scaler to encode to linear regression.

# Linear Regression Pipeline (Sklearn)

This project is a reusable machine learning pipeline for training and evaluating a **linear regression model(Single or Multiple**, designed to work with **any numerical dataset** (optionally with categorical features).

It includes:
- Automated **EDA (Exploratory Data Analysis)** with plots
- Preprocessing using `ColumnTransformer` (`SimpleImputer`,`StandardScaler`, `OneHotEncoder`)
- Model training, evaluation & Visualization (`MAPE`, actual vs predicted plots & Residual plots)
- Modular code for scalability and reuse

## 📂 Project Structure
- `eda.py`: Reusable EDA visualizations
- `train.py`: Handles preprocessing, training, and evaluation
- `main.py`: Entry point to run the pipeline
- `dataframe.py`: currently i have included two dataframes i.e. 50 startup dataframe and boston housing price prediction - will be adding more.
- `visualize.py` : created for plotting regression graphs i.e. actual vs. predicted scatter plot and residual plots. i will be adding more as and when i will be working with more and more databases
- `requirements.txt`: All dependencies

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/Linear-Regression-pipeline.git
cd linear-regression-pipeline

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py

