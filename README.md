# Feature-Engineering-Workshop


# 🛠️ Feature Engineering Workshop

This repository contains a hands-on implementation of **Feature Engineering** to improve machine learning model performance, demonstrated on the **California Housing Dataset**. The focus is on creating **polynomial and interaction features** to capture hidden patterns that a basic linear model might miss.


## 📌 **About Feature Engineering**

Feature engineering is the process of **transforming raw data into meaningful features** that better represent the underlying problem to the predictive model. It’s one of the most important steps in the machine learning pipeline because:

* Models can only learn from the information provided in features.
* Good features can improve accuracy without changing the model type.
* Feature interactions can reveal non-linear relationships.
* It reduces model bias and improves generalization.


## 🎯 **Project Objectives**

* **Understand baseline performance** of a simple linear regression model.
* **Apply polynomial & interaction features** to enhance input variables.
* **Compare performance metrics** to evaluate improvements.
  

## 📂 **Workflow**

1. **Data Loading**

   * Fetch California Housing Dataset from `sklearn.datasets`.
   * Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.
2. **Data Splitting**

   * 80% training data, 20% testing data.
3. **Baseline Model**

   * Train a simple **Linear Regression** model.
   * Record **RMSE** and **R² Score**.
4. **Feature Engineering**

   * Use `PolynomialFeatures` with:
    * **degree=2**
    * **interaction\_only=True**
   * Generate additional features that capture relationships between variables.
5. **Enhanced Model**

   * Train a new Linear Regression model using the engineered dataset.
6. **Performance Comparison**

   * Compare baseline vs engineered feature models.


## 📊 **Sample Results**

| Model                          | RMSE | R² Score |
| ------------------------------ | ---- | -------- |
| Baseline Linear Regression     | 0.73 | 0.60     |
| Polynomial & Interaction Model | 0.69 | 0.64     |

*Note: Results may vary slightly due to randomization.*


## 💡 **Key Learnings**

* Even simple models can benefit greatly from better features.
* Interaction features often reveal relationships hidden to basic models.
* RMSE and R² are essential for measuring regression model performance.


## 📌 **How to Run**

# Clone the repository
git clone https://github.com/yourusername/feature-engineering-workshop.git

# Navigate to the folder
cd feature-engineering-workshop

# Install dependencies
pip install pandas numpy scikit-learn

# Run the script
python feature_engineering.py


## 🔮 **Next Steps**

* Try `degree=3` polynomial features for more complexity.
* Test feature selection techniques to avoid overfitting.
* Explore scaling/normalization before applying polynomial transformations.
