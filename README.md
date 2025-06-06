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

# Elevate-labs-internship-task-4

## Objective
To build a binary classifier using Logistic Regression on a suitable dataset, evaluate its performance using various metrics, understand threshold tuning, and explain key machine learning concepts related to binary classification[cite: 1, 5].

## Tools Used
* Python
* Scikit-learn [cite: 2]
* Pandas [cite: 2]
* Matplotlib [cite: 2]

## Dataset
For this task, the Breast Cancer Wisconsin (Diagnostic) Dataset was used[cite: 5]. This dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses, used to predict whether a mass is benign (B) or malignant (M).

## Task Steps & Implementation

1.  **Dataset Loading and Preprocessing:**
    * The dataset was loaded using Pandas.
    * Irrelevant columns (like `Unnamed: 32` and `id`) were removed.
    * Missing values were dropped.
    * The `diagnosis` column was converted to a binary format: 'M' (Malignant) mapped to 1, and 'B' (Benign) mapped to 0.

2.  **Train/Test Split and Feature Standardization:**
    * The dataset was split into training and testing sets (80% training, 20% testing) to evaluate the model's performance on unseen data. `stratify=y` was used to maintain the proportion of classes in both sets.
    * `StandardScaler` from Scikit-learn was used to standardize the features[cite: 3], which is crucial for algorithms like Logistic Regression to ensure features contribute equally to the distance calculations.

3.  **Logistic Regression Model Training:**
    * A `LogisticRegression` model was initialized and trained on the scaled training data[cite: 3].

4.  **Model Evaluation:**
    * The model's performance was evaluated using:
        * **Confusion Matrix:** A table showing the counts of true positive, true negative, false positive, and false negative predictions[cite: 4, 7].
        * **Precision:** The proportion of true positive predictions among all positive predictions[cite: 4, 7].
        * **Recall:** The proportion of true positive predictions among all actual positive instances[cite: 4, 7].
        * **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve):** A metric that measures the overall performance of a binary classifier across all possible classification thresholds[cite: 4, 7]. An AUC of 1.0 indicates a perfect model.

### Evaluation Results (Default Threshold)

Confusion Matrix:
[[71  1]
[ 3 39]]
Precision: 0.97
Recall: 0.93
ROC-AUC: 1.00


5.  **ROC Curve:**
    * The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings. The area under this curve (AUC) provides an aggregate measure of performance[cite: 4, 7].

![image](https://github.com/user-attachments/assets/c5adcbda-49ec-4e5a-99d6-0b86b88eabf0)

6.  **Threshold Tuning:**
    * Logistic Regression models output probabilities. A threshold is used to convert these probabilities into binary class predictions. The default threshold is typically 0.5[cite: 4].
    * Tuning the threshold allows us to adjust the balance between precision and recall based on the specific needs of the problem[cite: 4, 8]. For instance, in medical diagnosis, it might be more critical to maximize recall (minimize false negatives) even if it means a slight reduction in precision.
    * An example of adjusting the threshold to 0.3 was demonstrated.

### Evaluation Results (Threshold = 0.3)

With threshold = 0.3:
Confusion Matrix:
[[71  1]
[ 1 41]]
Precision: 0.98
Recall: 0.98

*Note: Adjusting the threshold from 0.5 to 0.3 in this instance improved recall from 0.93 to 0.98, with a slight increase in precision, indicating a better identification of positive cases without significantly increasing false positives.*

7.  **Sigmoid Function:**
    * The sigmoid function (also known as the logistic function) is a key component of Logistic Regression[cite: 4, 6]. It maps any real-valued number to a value between 0 and 1, making it suitable for interpreting as a probability.
    * The formula for the sigmoid function is $ \sigma(z) = \frac{1}{1 + e^{-z}} $.
    * It produces an S-shaped curve, squashing the input values into a probability scale.

![image](https://github.com/user-attachments/assets/229f97f6-bc98-4b09-b367-764729f25a80)


## Files in this Repository
* `logistic_regression_classifier.py`: The Python script containing the code for data preprocessing, model training, evaluation, and plotting.
* `data.csv`: The Breast Cancer Wisconsin dataset used for this task.
* `roc_curve.png`: Image showing the ROC curve generated by the script.
* `sigmoid_curve.png`: Image showing the plot of the sigmoid function.
* `README.md`: This file.

---
**Disclaimer**: This project was completed as part of an AI & ML Internship task.



# Elevate-labs-internship-task-5

## Objective
The objective of this task was to learn and implement tree-based models for classification and regression, specifically focusing on Decision Trees and Random Forests. Key aspects included understanding model training, visualizing trees, analyzing overfitting, controlling model complexity, evaluating feature importance, and performing cross-validation.

## Tools Used
* Python
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn

## Dataset
The [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) was used for this task.

## Implementation Steps and Analysis

### 1. Data Loading and Splitting
The `heart.csv` dataset was loaded, and features (X) and the target variable (y) were separated. The data was then split into training and testing sets with a 70/30 ratio.

* **Training Data Shape:** (717, 13)
* **Testing Data Shape:** (308, 13)

### 2. Decision Tree Classifier Training and Visualization (Initial)
A Decision Tree Classifier was trained on the training data. The initial accuracy on the test set was calculated. The decision tree was then visualized using `matplotlib.pyplot.plot_tree`.

* **Decision Tree Accuracy (before depth control):** 0.9708

**Initial Decision Tree Visualization:**
![image](https://github.com/user-attachments/assets/66b5f309-93c9-4bc3-a0f9-3b9569c7f0a0)


### 3. Overfitting Analysis and Depth Control for Decision Tree
To understand and mitigate overfitting, a loop was run to train Decision Trees with varying `max_depth` values (from 1 to 14). The training and testing accuracies for each depth were recorded and plotted to identify an optimal depth where the model generalizes well without overfitting.

The analysis indicated an optimal depth for the Decision Tree model.

* **Optimal Decision Tree depth based on testing accuracy:** 10
* **Decision Tree Accuracy (optimal depth=10):** 0.9708

**Decision Tree Accuracy vs. Max Depth Plot:**
![image](https://github.com/user-attachments/assets/e3c0dbe5-837a-4b16-ae4d-3330e60c2d19)


**Optimal Decision Tree Visualization:**
![image](https://github.com/user-attachments/assets/2ca757d9-0878-4a35-b106-09cffa1df1dc)


### 4. Random Forest Classifier Training and Comparison
A Random Forest Classifier was trained with 100 estimators. Its accuracy on the test set was calculated and compared against the optimal Decision Tree's accuracy. Random Forests, being an ensemble method, typically offer better generalization and higher accuracy than single decision trees by reducing variance.

* **Random Forest Accuracy:** 0.9805
* **Comparison: Random Forest (0.9805) vs. Optimal Decision Tree (0.9708)**

### 5. Feature Importances (Random Forest)
The feature importances from the trained Random Forest model were extracted and visualized to understand which features contributed most significantly to the model's predictions.

**Random Forest Feature Importances Plot:**
![image](https://github.com/user-attachments/assets/ddfca7d4-e007-43c6-9bbc-9e99565ea35c)


### 6. Cross-Validation Evaluation
Both the optimal Decision Tree and the Random Forest models were evaluated using 5-fold cross-validation on the entire dataset to get a more robust estimate of their performance.

* **Decision Tree Cross-Validation Scores:** `[1. 1. 1. 1. 1.]`
* **Decision Tree Mean CV Accuracy:** 1.0000

* **Random Forest Cross-Validation Scores:** `[1. 1. 1. 1. 0.98536585]`
* **Random Forest Mean CV Accuracy:** 0.9971

## Conclusion
The task successfully demonstrated the implementation and evaluation of Decision Trees and Random Forests for classification. The Random Forest model generally showed better performance (higher accuracy) and robustness, as expected from an ensemble method, especially when considering the cross-validation results. Feature importance analysis provided insights into the most influential predictors in the dataset.



# Elevate-labs-internship-task-6

## Objective
This task aimed to understand and implement the K-Nearest Neighbors (KNN) algorithm for classification problems.

## Tools Used
* Python
* Scikit-learn (sklearn)
* Pandas
* Matplotlib
* Seaborn

## Dataset
The Iris dataset was used for this task, loaded from the provided `Iris.csv` file.
The dataset contains the following columns: `Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, and `Species`.

## Implementation Steps & Observations

### 1. Data Loading and Preprocessing
The `Iris.csv` file was loaded using Pandas. The 'Id' column was dropped as it's not a feature, and 'Species' was identified as the target variable. The `Species` column, containing categorical labels (e.g., 'Iris-setosa'), was converted into numerical format using `LabelEncoder`.

The dataset was then split into training (70%) and testing (30%) sets using `train_test_split`, ensuring stratification to maintain the class distribution in both sets.

**Normalization:** Given that KNN relies on distance calculations, feature scaling is crucial. `MinMaxScaler` was used to normalize the features, transforming them to a range between 0 and 1.

**Original Data Head:**
Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa


**Target Labels and Names:**
Target Labels (encoded): [0 1 2]
Original Target Names: ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']


**Scaled Data (first 5 rows of training set):**
[[0.22222222 0.20833333 0.32758621 0.41666667]
[0.52777778 0.08333333 0.5862069  0.58333333]
[0.22222222 0.75       0.06896552 0.08333333]
[0.69444444 0.5        0.82758621 0.91666667]
[0.38888889 0.33333333 0.51724138 0.5       ]]


### 2. Experimenting with Different K Values and Model Evaluation
The `KNeighborsClassifier` from scikit-learn was used. The model was trained and evaluated for various odd values of `K` ranging from 1 to 19. For each `K`, the model's performance was assessed using:
* **Accuracy Score:** The proportion of correctly classified instances.
* **Confusion Matrix:** A table showing the counts of true positive, true negative, false positive, and false negative predictions.

**Observations on Accuracy and Confusion Matrices:**

* **For K=1:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 15  0]
         [ 0  3 12]]
        ```
    * *Interpretation:* The model perfectly classified Iris-setosa and Iris-versicolor (15 each). However, for Iris-virginica, 3 instances were misclassified as Iris-versicolor, while 12 were correctly classified.

    ![image](https://github.com/user-attachments/assets/02417332-454c-4acb-9240-63d68052cd79)


* **For K=3:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 15  0]
         [ 0  3 12]]
        ```
    * *Interpretation:* Similar to K=1, the model correctly classified 15 Iris-setosa and 15 Iris-versicolor. 3 Iris-virginica were still misclassified as Iris-versicolor.

  ![image](https://github.com/user-attachments/assets/e0d59584-9441-475a-be89-d788d5f5f090)


* **For K=5:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 15  0]
         [ 0  3 12]]
        ```
    * *Interpretation:* Performance remains consistent with K=1 and K=3.

    **[Insert image for Confusion Matrix for K=5 here, e.g., `image_e274f9.png`]**

* **For K=7:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 15  0]
         [ 0  3 12]]
        ```
    * *Interpretation:* Accuracy and misclassifications are still the same.

   ![image](https://github.com/user-attachments/assets/09bbf150-8717-402f-8245-571e149a0096)


* **For K=9:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 15  0]
         [ 0  3 12]]
        ```
    * *Interpretation:* No change in accuracy or confusion matrix for K=9.

    ![image](https://github.com/user-attachments/assets/43560b3b-cfb2-40e5-966c-da9615279a17)


* **For K=11:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 15  0]
         [ 0  3 12]]
        ```
    * *Interpretation:* The model maintains the same accuracy and misclassification pattern up to K=11.

   ![image](https://github.com/user-attachments/assets/d436b4fa-5d9e-4024-a109-97edb7a76ab4)


* **For K=13:**
    * Accuracy: 0.9333
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 14  1]
         [ 0  2 13]]
        ```
    * *Interpretation:* The accuracy remains the same, but the misclassification pattern changes slightly: 1 Iris-versicolor was misclassified as Iris-virginica, and 2 Iris-virginica were misclassified as Iris-versicolor.

    ![image](https://github.com/user-attachments/assets/41d84ef1-0a1a-4ba5-bad9-6ccef94fa5c4)


* **For K=15:**
    * Accuracy: 0.9111
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 14  1]
         [ 0  3 12]]
        ```
    * *Interpretation:* The accuracy drops for K=15. Now, 1 Iris-versicolor is misclassified as Iris-virginica, and 3 Iris-virginica are misclassified as Iris-versicolor.

   ![image](https://github.com/user-attachments/assets/d4e4a4f1-3e30-4bf2-aede-05927b9e1162)

* **For K=17:**
    * Accuracy: 0.9111
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 14  1]
         [ 0  3 12]]
        ```
    * *Interpretation:* Accuracy and confusion matrix are consistent with K=15.

 ![image](https://github.com/user-attachments/assets/71464f44-7370-470b-801a-84680283b0ca)


* **For K=19:**
    * Accuracy: 0.9111
    * Confusion Matrix:
        ```
        [[15  0  0]
         [ 0 14  1]
         [ 0  3 12]]
        ```
    * *Interpretation:* Accuracy and confusion matrix are consistent with K=15 and K=17.

    ![image](https://github.com/user-attachments/assets/349ddabb-2af5-4842-b3b2-45e77366aba6)


**Accuracy vs. K Value Plot:**

This plot visually summarizes how accuracy changes with different K values.

![image](https://github.com/user-attachments/assets/9f432418-b9f7-49fe-b901-22ac68b761fb)


* *Observation:* The plot clearly shows that the highest accuracy was achieved for K values from 1 to 13 (inclusive), after which the accuracy starts to decrease. This indicates that for this particular dataset and split, a smaller K is generally better. The optimal K found by `np.argmax(accuracies)` was K=1.

### 3. Visualizing Decision Boundaries
To visualize decision boundaries, only two features (Petal Length and Petal Width) were selected, as this allows for a 2D plot. An optimal K (determined from the accuracy plot, which was K=1 in this case) was chosen to train the final KNN model for visualization. A mesh grid was created, and predictions were made across this grid to illustrate the regions classified by the KNN model. The original data points (training and test) were then overlaid on this plot.

**Decision Boundary Plot:**

![image](https://github.com/user-attachments/assets/68f94c74-89b7-42b4-845a-28eb36ef33fd)


* *Interpretation:* This plot shows how the KNN model divides the feature space into regions corresponding to each Iris species. The boundaries are non-linear and depend on the proximity to training data points. For K=1, the boundaries are very sensitive to individual data points.

## Conclusion
This task successfully demonstrated the implementation of the KNN algorithm for classification. Key steps included data loading, feature normalization (crucial for distance-based algorithms), splitting data, training the `KNeighborsClassifier` with varying `K` values, evaluating performance using accuracy and confusion matrices, and visualizing decision boundaries. The experiment showed that for the Iris dataset, a lower K value (like K=1 to K=13) yielded better or comparable accuracy, with K=1 providing the highest observed accuracy in this run.

## How to Run the Code
1.  Save the provided Python code (e.g., `knn_iris_classification.py`) and the `Iris.csv` file in the same directory.
2.  Ensure you have the necessary libraries installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`. If not, install them using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Run the Python script from your terminal:
    ```bash
    python knn_iris_classification.py
    ```
    Or, if you are using a Jupyter Notebook, run the cells sequentially.



    # Elevate-labs-internship-task-7
    ## Objective
This task focused on implementing Support Vector Machines (SVMs) for binary classification, exploring both linear and non-linear (RBF kernel) approaches. Key learning areas included margin maximization, the kernel trick, and hyperparameter tuning.

## Tools Used
* Python
* Scikit-learn (sklearn)
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Dataset
The Breast Cancer dataset, provided as `breast-cancer.csv`, was used for this task. This dataset is commonly used for binary classification, where the goal is to classify a tumor as either benign ('B') or malignant ('M') based on various features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

The dataset contains 32 columns, including an `id` column, `diagnosis` (target variable), and 30 numerical features representing characteristics of the cell nuclei.

## Implementation Steps & Observations

### 1. Data Loading and Preprocessing
The `breast-cancer.csv` file was loaded using Pandas. The `id` column was dropped as it's an identifier and not a feature. The `diagnosis` column, which contains categorical labels ('M' for Malignant, 'B' for Benign), was encoded into numerical format (typically 0 for 'B' and 1 for 'M') using `LabelEncoder`.

The dataset was then split into training (70%) and testing (30%) sets using `train_test_split`, ensuring stratification to maintain the original class distribution in both sets.

**Feature Scaling:** SVMs are highly sensitive to the scale of features. Therefore, `StandardScaler` was applied to normalize the feature data. This transforms the data to have a mean of 0 and a standard deviation of 1.

**Original Data Head:**
   id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  ...  symmetry_worst  fractal_dimension_worst  
0   842302         M        17.99         10.38          122.80     1001.0  ...          0.4601                  0.11890

1   842517         M        20.57         17.77          132.90     1326.0  ...          0.2750                  0.08902

2  84300903         M        19.69         21.25          130.00     1203.0  ...          0.3613                  0.08758

3  84348301         M        11.42         20.38           77.58      386.1  ...          0.6638                  0.17300

4  84358402         M        20.29         14.34          135.10     1297.0  ...          0.2364                  0.07678

[5 rows x 32 columns]


**DataFrame Info:**
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 32 columns):

Column Non-Null Count Dtype
0   id                       569 non-null    int64

1   diagnosis                569 non-null    object
2   radius_mean              569 non-null    float64
3   texture_mean             569 non-null    float64
4   perimeter_mean           569 non-null    float64
5   area_mean                569 non-null    float64
6   smoothness_mean          569 non-null    float64
7   compactness_mean         569 non-null    float64
8   concavity_mean           569 non-null    float64
9   concave points_mean      569 non-null    float64
10  symmetry_mean            569 non-null    float64
11  fractal_dimension_mean   569 non-null    float64
12  radius_se                569 non-null    float64
13  texture_se               569 non-null    float64
14  perimeter_se             569 non-null    float64
15  area_se                  569 non-null    float64
16  smoothness_se            569 non-null    float64
17  compactness_se           569 non-null    float64
18  concavity_se             569 non-null    float64
19  concave points_se        569 non-null    float64
20  symmetry_se              569 non-null    float64
21  fractal_dimension_se     569 non-null    float64
22  radius_worst             569 non-null    float64
23  texture_worst            569 non-null    float64
24  perimeter_worst          569 non-null    float64
25  area_worst               569 non-null    float64
26  smoothness_worst         569 non-null    float64
27  compactness_worst        569 non-null    float64
28  concavity_worst          569 non-null    float64
29  concave points_worst     569 non-null    float64
30  symmetry_worst           569 non-null    float64
31  fractal_dimension_worst  569 non-null    float64
dtypes: float64(30), int64(1), object(1)
memory usage: 142.4+ KB


**Target Labels and Names:**
Target Labels (encoded): [0 1]
Original Target Names: ['B' 'M']
Shape of X: (569, 30)
Shape of y: (569,)


**Scaled Data (first 5 rows of training set):**
[[ 1.70581999  1.04974466  2.12129465  1.66933676  2.27645789  4.54429486
3.53913939  2.84839268  4.03238962  2.63977264  1.88658218  1.20005314
2.87275048  1.74921741  1.19777257  2.41558743  2.3247057   2.29784843
6.94476933  0.82740765  1.45951037  0.94958949  1.86817555  1.3068397
1.33538327  2.31385505  2.34642696  2.04939907  3.94428119  0.86655367]
[ 0.53638811 -0.90994477  0.5690952   0.39031555 -0.06586788  0.64494486
0.39354254  0.55700173 -0.11678098 -0.40152959 -0.25041371 -0.84442436
-0.11857378 -0.16149227 -0.37614552  0.40854639  0.14771626  0.48165509
-0.68243869 -0.23809862  0.50829368 -0.64531469  0.50191659  0.33008672
0.47701253  1.24209126  1.05257355  1.25584994  0.16797759  0.40524244]
[-0.13145612 -0.99333581 -0.14884823 -0.21154347 -0.91616898 -0.50857386
-0.56752982 -0.60664497 -0.07986666 -0.51245428 -0.95807347 -1.2145298
-0.87783349 -0.66531468 -0.59346476  0.26263059 -0.23715248 -0.51677077
-0.26595611  0.41139177 -0.36245131 -1.02848469 -0.28041517 -0.39938703
-0.19621941  0.36866098 -0.08079455 -0.14001353  0.74198821  0.68752098]
[ 0.92807734  1.41805508  0.92540484  0.82854593  0.32973419  0.22374766
0.98318775  0.44394722 -1.86652019 -0.55598169  0.44143242  0.80819795
0.44689582  0.41909869  8.38975683  3.37717812  3.37462278  4.43435806
0.15764583  3.27108269  0.36592069  0.35190979  0.38608159  0.24353898
-0.3308658  -0.46472119  0.03031066 -0.07225317 -2.04071547 -0.86116674]
[ 0.14469888 -0.97248805  0.15831525 -0.00969706  1.20575985  0.58504125
0.13770066  0.56285816  1.11615761  0.1179911   0.37638091 -0.86311837
0.47025002  0.05663157 -0.49087067  1.06097449  0.36586906  0.74431445
0.45665897  0.40135385  0.00895651 -1.2577317   0.05936749 -0.14045035
-0.18780401  0.35912571 -0.06417625  0.37647101  0.3738727   0.04772623]]


### 2. Training SVM with Linear Kernel
A Support Vector Classifier (`SVC`) was trained using a linear kernel.

**Linear SVM Evaluation:**
* **Accuracy:** 0.9649
* **Classification Report:**
    ```
                  precision    recall  f1-score   support

            B         0.95      1.00      0.97       107
            M         1.00      0.91      0.95        64

     accuracy                           0.96       171
    macro avg     0.97      0.95      0.96       171
 weighted avg     0.97      0.96      0.96       171
    ```
* **Confusion Matrix:**
    ```
    [[107   0]
     [  6  58]]
    ```
    *Interpretation:* The linear SVM model correctly classified all 107 benign cases (B). For malignant cases (M), it correctly identified 58 but misclassified 6 as benign.

**Confusion Matrix for Linear SVM:**
![image](https://github.com/user-attachments/assets/74b5134c-e9ba-4991-92d6-f6568bf02a39)


### 3. Training SVM with RBF Kernel
An `SVC` was also trained using the Radial Basis Function (RBF) kernel, which is suitable for non-linear decision boundaries.

**RBF SVM Evaluation (Default Hyperparameters):**
* **Accuracy:** 0.9591
* **Classification Report:**
    ```
                  precision    recall  f1-score   support

            B         0.94      1.00      0.97       107
            M         1.00      0.89      0.94        64

     accuracy                           0.96       171
    macro avg     0.97      0.95      0.96       171
 weighted avg     0.96      0.96      0.96       171
    ```
* **Confusion Matrix:**
    ```
    [[107   0]
     [  7  57]]
    ```
    *Interpretation:* The RBF SVM, with default parameters, also classified all 107 benign cases correctly. It misclassified 7 malignant cases as benign, which is slightly more than the linear kernel.

**Confusion Matrix for RBF SVM (Default):**
![image](https://github.com/user-attachments/assets/c2d7bc34-1f4e-4008-9221-403d1cb527df)


### 4. Visualize Decision Boundary (2D)
To visualize the decision boundary, two key features (`mean radius` and `mean texture`) were selected from the dataset. A separate RBF SVM model was trained on these two scaled features, and its decision boundary was plotted over a mesh grid, with training and test data points overlaid.

**SVM RBF Kernel Decision Boundary (2 Features):**
![image](https://github.com/user-attachments/assets/4eab7c9b-295f-4e32-b2db-04771760fcc7)

*Interpretation:* The plot illustrates how the RBF kernel creates a non-linear boundary to separate the two classes ('B' and 'M') based on the chosen features. The colored regions represent the areas where the model predicts 'B' or 'M'.

### 5. Hyperparameter Tuning with GridSearchCV
To optimize the performance of the RBF kernel SVM, `GridSearchCV` was used to find the best combination of `C` (regularization parameter) and `gamma` (kernel coefficient). Cross-validation (5-fold) was inherently part of this process.

**Hyperparameter Tuning Results:**
Fitting 5 folds for each of 16 candidates, totalling 80 fits

Best parameters found: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
Best cross-validation accuracy: 0.9698


**Tuned RBF SVM Evaluation (on Test Set):**
* **Accuracy:** 0.9532
* **Classification Report:**
    ```
                  precision    recall  f1-score   support

            B         0.93      1.00      0.96       107
            M         1.00      0.88      0.93        64

     accuracy                           0.95       171
    macro avg     0.97      0.94      0.95       171
 weighted avg     0.96      0.95      0.95       171
    ```
* **Confusion Matrix:**
    ```
    [[107   0]
     [  8  56]]
    ```
    *Interpretation:* While the cross-validation accuracy was very high (0.9698), the test set accuracy slightly decreased to 0.9532. The model misclassified 8 malignant cases as benign, which is a slight increase compared to the RBF SVM with default parameters. This highlights that while cross-validation gives a good estimate, test set performance can vary.

**Confusion Matrix for Tuned RBF SVM:**
![image](https://github.com/user-attachments/assets/42a65989-7150-4710-b27c-2ffcf951050d)


### 6. Cross-validation for the Best SVM Model
To further confirm the stability of the chosen model, cross-validation was performed directly on the best `SVC` estimator found by `GridSearchCV`.

**Cross-validation Scores:**
Cross-validation accuracies (5 folds): [0.975        0.975        0.9875       0.97468354 0.93670886]
Mean cross-validation accuracy: 0.9698
Standard deviation of cross-validation accuracies: 0.0172

*Interpretation:* The mean cross-validation accuracy of ~0.9698 confirms the model's robust performance across different folds of the training data. The relatively low standard deviation (0.0172) indicates consistency in performance.

## Conclusion
This task successfully demonstrated the application of Support Vector Machines for binary classification on the Breast Cancer dataset. We explored both linear and RBF kernels, understanding their performance differences. The importance of feature scaling for SVMs was highlighted. Hyperparameter tuning using `GridSearchCV` and cross-validation proved crucial for finding optimal model configurations and assessing their generalization capability. While the linear kernel performed slightly better on the test set in this specific run, the RBF kernel with tuned hyperparameters also showed strong performance and is capable of capturing non-linear relationships.

## How to Run the Code
1.  Save the provided Python code (e.g., `svm_breast_cancer.py`) and the `breast-cancer.csv` file in the same directory.
2.  Ensure you have the necessary libraries installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`. If not, install them using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Run the Python script from your terminal:
    ```bash
    python svm_breast_cancer.py
    ```
    Or, if you are using a Jupyter Notebook, run the cells sequentially.
    

# Elevate-labs-internship-task-8

## Objective
This task focuses on performing unsupervised learning using the K-Means clustering algorithm. The primary goal is to segment customers based on their annual income and spending score, identify the optimal number of clusters using the Elbow Method, and evaluate the clustering quality using the Silhouette Score.

## Tools Used
* Python
* Scikit-learn (`sklearn`)
* Pandas
* Matplotlib
* NumPy

## Dataset
For this task, the **Mall Customer Segmentation Dataset** (`Mall_Customers.csv`) was utilized. This dataset contains customer information, including `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The objective is to group customers into distinct segments based on their `Annual Income (k$)` and `Spending Score (1-100)` to aid in targeted marketing strategies.

## Implementation Steps & Observations

### 1. Data Loading and Preprocessing
The `Mall_Customers.csv` dataset was loaded into a Pandas DataFrame. For clustering, the `Annual Income (k$)` and `Spending Score (1-100)` columns were selected as features. The `CustomerID` column was excluded as it's an identifier, and `Gender` was not included in this specific clustering analysis to focus on the income-spending relationship. The selected features were then scaled using `StandardScaler` to ensure that features with larger values do not dominate the distance calculations in K-Means.

**Original Data Head:**
CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40


**DataFrame Info:**
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):

Column Non-Null Count Dtype
0   CustomerID              200 non-null    int64
1   Gender                  200 non-null    object
2   Age                     200 non-null    int64
3   Annual Income (k$)      200 non-null    int64
4   Spending Score (1-100)  200 non-null    int64
dtypes: int64(4), object(1)
memory usage: 7.9+ KB


**Scaled Features (first 5 rows):**
[[-1.73899919 -0.43480148]
[-1.73899919  1.19570407]
[-1.70082976 -1.71591298]
[-1.70082976  1.04041783]
[-1.66266033 -0.39597992]]


### 2. Elbow Method to Find Optimal K
The Elbow Method was employed to determine the optimal number of clusters (K). This involves fitting K-Means for a range of K values (e.g., 1 to 10) and calculating the inertia (sum of squared distances of samples to their closest cluster center) for each K. The "elbow point" in the plot of inertia vs. K indicates the point where the decrease in inertia starts to slow down significantly, suggesting an appropriate number of clusters.

**Elbow Method Results (Inertia for K=1 to 10):**
K=1: Inertia=400.00
K=2: Inertia=269.69
K=3: Inertia=157.70
K=4: Inertia=108.92
K=5: Inertia=65.57
K=6: Inertia=55.06
K=7: Inertia=44.86
K=8: Inertia=37.23
K=9: Inertia=32.39
K=10: Inertia=29.98


**Elbow Method Plot:**
![image](https://github.com/user-attachments/assets/5aa3d3f0-4316-4717-a6b7-813336f745d7)

*Interpretation:* From the plot, the 'elbow' appears to be at K=5, indicating that 5 is a reasonable number of clusters for this dataset, as adding more clusters beyond this point provides diminishing returns in terms of reducing inertia.

### 3. K-Means Clustering
With the optimal K determined as 5, the K-Means algorithm was applied to the scaled features. Each customer was then assigned a cluster label.

**Number of points per cluster (for K=5):**
Cluster
0     81
1     39
2     22
3     35
4     23
Name: count, dtype: int64


### 4. Cluster Visualization
The segmented customers were visualized on a scatter plot, with 'Scaled Annual Income (k$)' on the x-axis and 'Scaled Spending Score (1-100)' on the y-axis. Each customer point was color-coded according to its assigned cluster, and the cluster centroids were marked.

**K-Means Clustering Plot (K=5):**
![image](https://github.com/user-attachments/assets/9e83ec3a-325d-4622-8096-d18401136582)

*Interpretation:* The visualization clearly shows 5 distinct customer segments based on their income and spending habits. These segments could represent different customer profiles (e.g., high income-high spending, low income-low spending, etc.).

### 5. Evaluate Clustering Using Silhouette Score
The Silhouette Score was used to evaluate the quality of the clustering. This metric measures how similar an object is to its own cluster compared to other clusters. A higher Silhouette Score indicates better-defined clusters.

**Silhouette Score for K=5:**
Silhouette Score: 0.555

*Interpretation:* A Silhouette Score of approximately 0.555 indicates that the clusters are reasonably well-separated, and customers within each cluster are relatively similar to each other.


## How to Run the Code
1.  Save the Python code (e.g., `kmeans_customer_segmentation.py`) and the `Mall_Customers.csv` file in the same directory.
2.  Ensure you have the necessary libraries installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`. If not, install them using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  Run the Python script from your terminal:
    ```bash
    python kmeans_customer_segmentation.py
    ```
    Or, if you are using a Jupyter Notebook, run the cells sequentially.




  
