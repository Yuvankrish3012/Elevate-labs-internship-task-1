# Elevate-labs-internship-task-1

# Titanic Data Cleaning & Preprocessing 
![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)

This repository contains my solution for **Task 1: Data Cleaning & Preprocessing** of the Titanic dataset as part of my AI/ML internship with Elevate Labs.

## 📋 Project Overview
This project demonstrates fundamental data preprocessing techniques applied to the classic Titanic dataset to prepare it for machine learning models. The notebook covers:
- Missing value treatment
- Feature engineering
- Categorical encoding
- Outlier handling
- Feature scaling

## 🛠️ Technical Implementation
### Libraries Used
- Python 3.x
- Pandas (Data manipulation)
- NumPy (Numerical operations)
- Matplotlib & Seaborn (Visualizations)
- Scikit-learn (StandardScaler)

### Key Steps Performed
1. **Data Loading & Initial Exploration**
   - Checked dataset structure
   - Identified missing values

2. **Missing Value Treatment**
   - Filled missing `Age` with median
   - Filled missing `Embarked` with mode
   - Dropped `Cabin` column (too many missing values)

3. **Feature Engineering**
   - Created `FamilySize` (SibSp + Parch + 1)
   - Created `IsAlone` flag
   - Extracted `Title` from names (Mr, Mrs, etc.)

4. **Data Transformation**
   - Encoded categorical variables (Sex, Embarked, Title)
   - Scaled numerical features (Age, Fare, FamilySize)

5. **Outlier Handling**
   - Detected outliers using boxplots
   - Capped extreme `Fare` values

## 🚀 How to Use This Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ElevateLabs-Internship-Task1.git
   
# Elevate-labs-internship-task-2


## Task Objective
The objective of this task is to understand the provided dataset using statistical analysis and various visualizations. This involves exploring the data to identify patterns, trends, or anomalies, and making basic feature-level inferences from the visuals.

**Tools Used:** Pandas, Matplotlib, Seaborn

## Exploratory Data Analysis and Visualization Analysis

The following steps were performed for the Exploratory Data Analysis (EDA) on the Titanic Dataset, with analysis based on the provided images:

### 1. Initial Data Overview (Histograms of Numerical Features)

![image](https://github.com/user-attachments/assets/1243f02f-9850-416c-a462-4d6d18cb2e4d)


This image displays histograms for all numerical columns in the dataset: `PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, and `Fare`.

* **`PassengerId`**: Shows a uniform distribution, as expected for unique identifiers.
* **`Survived`**: Clearly indicates an imbalanced dataset, with significantly more passengers not surviving (0) than surviving (1).
* **`Pclass`**: Shows that `Pclass 3` has the highest number of passengers, followed by `Pclass 1` and then `Pclass 2`.
* **`Age`**: Reveals a right-skewed distribution, with a higher concentration of younger passengers (around 20-30 years old) and fewer older passengers.
* **`SibSp` (Siblings/Spouses Aboard)**: The majority of passengers traveled alone or with one sibling/spouse. Very few traveled with more than two.
* **`Parch` (Parents/Children Aboard)**: Similar to `SibSp`, most passengers traveled without parents or children, or with one or two.
* **`Fare`**: Highly right-skewed, indicating that most passengers paid lower fares, with a few outliers paying very high fares.

### 2. Age Distribution by Passenger Class (Boxplot)

![image](https://github.com/user-attachments/assets/af858921-71cf-417c-a6ff-15f8356407e0)


This boxplot visualizes the distribution of `Age` across different `Pclass` categories.

* **`Pclass 1`**: Passengers in first class tend to be older, with a median age around 37-38. The age range is wider, but outliers extend to around 80.
* **`Pclass 2`**: The median age for second class passengers is slightly lower than first class, around 29-30. The age distribution is also narrower than first class.
* **`Pclass 3`**: Passengers in third class are generally the youngest, with a median age around 24-25. This class also shows a significant number of outliers, including very young children.
* **Overall**: There's a clear trend: as passenger class decreases (from 1 to 3), the median age of passengers also tends to decrease. This suggests that wealthier (first-class) passengers were generally older.

### 3. Correlation Matrix (Heatmap)

![image](https://github.com/user-attachments/assets/f1b83277-ebe1-4c7d-872d-b636111b6bae)


This heatmap displays the correlation coefficients between all numerical features. The color intensity and value indicate the strength and direction of the correlation.

* **`Survived` vs. `Pclass`**: A strong negative correlation (-0.34) indicates that passengers in lower classes (higher `Pclass` number) had a lower chance of survival.
* **`Survived` vs. `Fare`**: A positive correlation (0.26) suggests that passengers who paid higher fares had a better chance of survival.
* **`Pclass` vs. `Fare`**: A strong negative correlation (-0.55) is observed, meaning higher passenger classes (lower `Pclass` number) are associated with higher fares.
* **`Age` vs. `Pclass`**: A negative correlation (-0.37) implies that older passengers tended to be in higher classes.
* **`SibSp` and `Parch`**: These two features show a moderate positive correlation (0.41), which is expected as they both relate to family size.
* **Other correlations**: Most other correlations are weak, indicating less direct linear relationships.

### 4. Pairplot of Key Features (Age, Fare, Pclass, Survived)

![image](https://github.com/user-attachments/assets/b9b5ff34-bd1f-4922-b27c-70e5dfe0967b)


This pairplot shows scatter plots for every combination of `Age`, `Fare`, and `Pclass`, with points colored by `Survived` status (0 for not survived, 1 for survived). The diagonal plots show kernel density estimates (KDEs) for each feature, split by survival status.

* **`Age` Distribution by Survival**: The KDE for `Age` on the diagonal shows that the distribution of ages for survivors (orange) is slightly shifted towards younger ages compared to non-survivors (blue), although there's significant overlap.
* **`Fare` Distribution by Survival**: The KDE for `Fare` on the diagonal clearly indicates that survivors (orange) tend to have paid higher fares, with a much wider spread towards higher values, while non-survivors (blue) are heavily concentrated at lower fares.
* **`Pclass` Distribution by Survival**: The KDE for `Pclass` on the diagonal strongly highlights that `Pclass 1` has a higher proportion of survivors, while `Pclass 3` has a significantly higher proportion of non-survivors.
* **`Age` vs. `Fare`**: The scatter plot shows a general trend where higher fares are paid by a wider range of ages, and survivors are more prevalent among those who paid higher fares.
* **`Age` vs. `Pclass`**: The scatter plot reinforces that `Pclass 1` (bottom row of `Pclass` values) has older passengers, and a higher survival rate is observed in this group. `Pclass 3` (top row of `Pclass` values) has younger passengers and a lower survival rate.
* **`Fare` vs. `Pclass`**: The scatter plot shows distinct clusters for each `Pclass`. `Pclass 1` has the highest fares, `Pclass 2` has moderate fares, and `Pclass 3` has the lowest fares. Survival is more common in the higher fare/lower `Pclass` clusters.
* **Overall**: The pairplot effectively visualizes the interplay between these key features and their impact on survival, reinforcing that `Fare` and `Pclass` are strong indicators of survival, and `Age` plays a role, particularly in conjunction with `Pclass`.

  # Elevate-labs-internship-task-3

  # 🏠 Housing Price Prediction using Linear Regression

This project implements both **Simple** and **Multiple Linear Regression** to predict house prices using various features from the housing dataset. It is done as part of the AI & ML Internship Task 3.

---

## 📦 Dataset

The dataset used is `Housing.csv`, which contains features such as:

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Road access, guestroom, basement, etc. (categorical features)

---

## 🧠 Problem Statement

To build a regression model that can predict house prices based on various features. The goal is to understand:

- How linear regression works
- Model evaluation metrics
- Feature impact on house pricing

---

## ⚙️ Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🔍 Workflow

### ✔️ 1. Data Preprocessing

- Loaded dataset using pandas
- Applied one-hot encoding to categorical variables
- Splitted into features (X) and target (y)
- Train-test split (80%-20%)

### ✔️ 2. Model Training

- Linear Regression using `sklearn.linear_model.LinearRegression`
- Trained on training set
- Tested on the test set

### ✔️ 3. Evaluation Metrics

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | ₹ 970,043.40 |
| **Mean Squared Error (MSE)** | ₹ 1,754,318,687,330.66 |
| **R² Score** | **0.6529** |

---

## 📈 Feature Coefficients

| Feature                       | Coefficient     |
|------------------------------|-----------------|
| area                         | ₹ 235.97 per sq.ft |
| bedrooms                     | ₹ 76,778.70 |
| bathrooms                    | ₹ 1,094,445.00 |
| stories                      | ₹ 407,476.60 |
| parking                      | ₹ 224,841.90 |
| mainroad_yes                 | ₹ 367,919.90 |
| guestroom_yes                | ₹ 231,610.00 |
| basement_yes                 | ₹ 390,251.20 |
| hotwaterheating_yes          | ₹ 684,649.90 |
| airconditioning_yes          | ₹ 791,426.70 |
| prefarea_yes                 | ₹ 629,890.60 |
| furnishingstatus_semi-furnished | ₹ -126,881.80 |
| furnishingstatus_unfurnished | ₹ -413,645.10 |

---

## 📊 Visualizations

### ✅ Actual vs Predicted Prices

![image](https://github.com/user-attachments/assets/21fd23f9-a99e-4d09-a828-e206d456e0a5)


> This plot shows how close the model's predicted prices are to the actual prices. A perfect model would align all points on a 45-degree line.





  
