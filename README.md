⚡ Electric Load Forecasting using a Hybrid ML/DL Model

This project aims to forecast short-term electric load demand using a hybrid approach that combines the strengths of deep learning and machine learning techniques. By leveraging time-series weather and load data, the system predicts future electricity consumption with enhanced accuracy, which can support utility providers in energy planning and grid optimization.

---

 📊 Dataset Overview

The project utilizes two primary datasets:
- **Electric Load Data:** 5-minute interval resolution
- **Weather Data:** 6-hour interval resolution (interpolated to 5-minute frequency)

 🗂️ Data Sources
- Publicly available datasets or utility-provided load/weather data (SLDC and Time and date websites)
- Data range: 2020 to 2022 
- If You want to skip the dataset creation process while training you can download the dataset directly and start from the training process
---

⚙️ Workflow

The forecasting pipeline includes:

1. **Data Collection**
   - Load and weather data scraped or downloaded
2. **Data Interpolation**
   - Upsampling weather data from 6-hour to 5-minute resolution
3. **Time-Series Resampling & Merging**
   - Synchronizing datasets for uniform frequency
4. **Exploratory Data Analysis (EDA)**
   - Visualization and statistical insights
5. **Feature Engineering**
   - Lag features, rolling stats, time-based features, etc.
6. **Train/Validation/Test Split**
7. **Model Construction**
   - A hybrid of deep learning and machine learning models
8. **Model Evaluation**
   - Metrics like MAE, RMSE, MAPE
98 **Result Visualization**

---

🧠 Model Highlights

- Hybrid architecture combining temporal pattern extraction (DL) and gradient-based boosting (ML)
- Scaled features for better convergence
- Robust evaluation over unseen data
- Window-based forecasting for capturing trends and periodicity

---

📦 Setup & Installation

### Prerequisites

- Python 3.7+
- Install dependencies:
  --pip install -r requirements.txt  ( SOME OF THE REQUIREMENTS MAYBE MISSING IN requirement.txt )

** How to Train the model ?

If u dont want to go through all the preprocessing Steps Just Download the dataset and run the HybridModelCreation.ipynb 
Use google colab or Jupiternotebook 
 
