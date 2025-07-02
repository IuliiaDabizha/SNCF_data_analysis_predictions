# 🚆 **Train Data Analysis Project**

This project is part of the **Epitech AI & Analytics curriculum**. It analyzes train journey data to understand patterns in **delays**, **station performance**, and **overall service quality**.
Additionally, it includes an **interactive dashboard** for visualizing key insights and predictions.

---

## 📂 **Project Structure**

```plaintext
.
├── 📄 dataset.csv                   # Dataset for compilation
├── 📄 README.md                     # Project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📓 tardis_eda.ipynb              # Exploratory Data Analysis notebook
├── 📓 tardis_model.ipynb            # Model training and evaluation notebook
└── 📄 tardis_dashboard.py           # Dashboard implementation script
```

## ✨ **Project Workflow**

### 1. 🧹 Data Cleaning and Preprocessing

- **Functions**: [`clean_data`](#), [`cap_and_clean_delay_outliers`](#)
- Handles missing values, standardizes station names, and caps outliers in delay data.

### 2. 📊 Exploratory Data Analysis (EDA)

- **Functions**: [`perform_eda`](#), [`delay_by_day`](#)
- Analyzes delay patterns, station performance, and time series trends.

### 3. 🤖 Model Development

- **Notebook**: [`tardis_model.ipynb`](#)
- Trains machine learning models to predict delays and evaluates their performance.

### 4. 📈 Visualization Dashboard

- **File**: [`tardis_dashboard.py`](#)
- Interactive dashboard for visualizing:
  - Delay distributions
  - Station performance rankings
  - Predictions for arrival times and delays

## ⚙️ **Installation and Setup**

### 1. **Clone the repository**

```bash
git clone git@github.com:EpitechPGEPromo2029/G-AIA-210-LYN-2-1-tardis-iuliia.dabizha.git
cd G-AIA-210-LYN-2-1-tardis-iuliia.dabizha
```

### 2. **Create a virtual environment**

```bash
python3 -m venv .venv

# On Linux and MacOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 **Usage**

### 1. **Run Jupyter Notebooks**

- Open and run [`tardis_eda.ipynb`](tardis_eda.ipynb) for exploratory data analysis.
- Open and run [`tardis_model.ipynb`](tardis_model.ipynb) for model training and evaluation.

### 2. **Run the Dashboard**

- Execute the following command to launch the dashboard:

     ```bash
     streamlit run tardis_dashboard.py
     ```

- Access the dashboard in your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 3. **View Results**

- Analysis results and visualizations are saved in the [`visualizations/`](visualizations/) directory.
- Key visualizations include:
  - Delay distributions
  - Top delayed stations
  - Trains scheduled over time

## 🛠️ **Project Features**

### 1. 🧹 Data Cleaning

- **Standardization**: Ensured consistency in station names.
- **Missing Values**: Imputed or removed incomplete records.
- **Outliers**: Capped and cleaned delay data for better model performance.

### 2. 📊 Exploratory Data Analysis (EDA)

- **Analysis**: Explored delay patterns, station performance, and time trends.
- **Visualizations**: Created charts for delay distributions, station rankings, and feature correlations.
- **Insights**: Identified key findings to guide model development.

### 3. 🤖 Model Training

- **Training**: Built machine learning models to predict train delays.
- **Evaluation**: Assessed models using accuracy, precision, and recall.
- **Deployment**: Saved the best model and preprocessing pipeline.

### 4. 📈 Dashboard Development

- **Visualization**: Designed an interactive dashboard for insights and predictions.
- **Features**: Added filtering by date, station, and delay type.
- **Real-Time Predictions**: Enabled delay predictions based on user input.

## 📦 **Required Libraries**

The project uses the following main libraries:

- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- fuzzywuzzy>=0.18.0
- python-Levenshtein>=0.21.0
- scikit-learn>=1.2.0
- streamlit>=1.45.1
- joblib

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

## ℹ️ **Project Information**

This project was developed as part of the **Epitech AI & Analytics curriculum**. It demonstrates the practical application of **data analysis**, **machine learning**, and **visualization techniques** in a real-world context. The focus is on analyzing train journey data to uncover patterns in delays, station performance, and overall service quality. Additionally, it includes an interactive dashboard for visualizing key insights and predictions.

---

## 📚 Tutorials & Explanations

### 🔍 Data Cleaning Explained

- We use [`fuzzywuzzy`](requirements.txt) to correct inconsistent station names (e.g., "PARlS" → "PARIS") by comparing string similarity.
- Delay outliers are capped to prevent them from skewing the model.
- Missing values are handled using imputation or row removal, depending on context.

### 📊 What is RMSE and Why Use It?

- **Root Mean Squared Error (RMSE)** measures the average error in our delay predictions.
- It gives more weight to large errors, making it a good fit for evaluating models where large prediction mistakes are costly (e.g., late trains).

### 🌲 Why We Use Random Forest?

- Random Forest is a robust ensemble learning method that combines many decision trees.
- It handles both categorical and numerical features well.
- It is robust to missing data and noise.
- It reduces overfitting by averaging many trees (ensemble learning).
- It can rank features by importance.

### 🪐 Jupyter Notebook

- #### <u>**This notebook is used to explore, clean, and prepare the dataset so it's ready for modeling.**</u>

  #### 1. Open the project folder

  - In VS Code, go to File > Open Folder...
  - Select the folder containing the [`.ipynb`](#) files (like tardis_eda.ipynb).

  #### 2. Create the Virtual Environment

  - Press [`Ctrl + Shift + P`](#) to open command palette.
  - Type: [`Python: Create Environment`](#) → press Enter.
  - Choose:
    - Environment  type: [`.venv`](#).
    - Enter interpreter path: The one with side note [`Global`](#).
  -💡 VS Code will create and configure your .venv automatically.

  #### 3. Open the Notebook

  - In the Explorer sidebar, click on any [`.ipynb`](tardis_model.ipynb) file (like tardis_model.ipynb).
  - The file will open in Jupyter view inside VS Code.

  #### 4. Select the Python Kernel

  - Top-right of the notebook → click the dropdown next to the current kernel
  - Choose: .venv or your active Python environment

  #### 5. Run the Notebook

  - In the toolbar above the notebook, click [`Run All`](#) to execute every cell
  - Or press [`▶️`](#) on individual cells / use Shift + Enter
  - Repeat this with another .ipynb file (like tardis_model.ipynb) if needed

## ✅ Deliverables Summary

You must submit the following files:

- [`tardis_eda.ipynb`](tardis_eda.ipynb) – A Jupyter Notebook that cleans and analyzes the data — it reads [`dataset.csv`](dataset.csv) and saves a cleaned version as [`cleaned_dataset.csv.`](cleaned_dataset.csv).
- [`tardis_model.ipynb`](tardis_model.ipynb) – A Jupyter Notebook that trains the machine learning model.
- [`tardis_dashboard.py`](tardis_dashboard.py) – A Python script that runs an interactive Streamlit dashboard.
- [`requirements.txt`](requirements.txt) – A file that lists everything your Python project needs to run.
- [`README.md`](README.md) – (This file) A clear file that explains how the project works, how to install it, and how to use it.

## 🧪 Evaluation Metrics

We evaluate models using:

- **RMSE** – Root Mean Squared Error (lower is better)
- **R² Score** – Explains how much variance is captured by the model
- **Model Comparison** – Justification of model choice based on scores and interpretability

---

<div align="center">

## 🙏 **Thanks**

***Special thanks to:***

**The Epitech AER** for their guidance and support.
**My fellow teammates** who helped test and provide valuable feedback.

</div>
