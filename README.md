# Fruits Prediction Project

This project focuses on building predictive models to classify different types of fruits based on their physical attributes. The primary objective is to explore and implement machine learning techniques to accurately predict the type of fruit using the provided dataset.

![Fruit Types](https://github.com/azamatgalidenov/fruitsPrediction/blob/main/img/fruits_type.png)

## Research Question

How accurately can we classify different fruits based on their physical attributes using machine learning models?

## Methodology

1. **Data Description**
   - **Dataset**: The dataset contains information about various fruits, including their type, color, size, and weight.
   - **Columns**:
     - `fruit_type`: The type of fruit. Includes categories such as 'Banana', 'Apple', and 'Grape'.
     - `color`: The color of the fruit. Categories include 'Yellow', 'Pink', 'Pale Yellow', 'Red', 'Creamy White', 'Green', 'Purple', and 'Black'.
     - `size`: The size category of the fruit. Options are 'Tiny', 'Large', 'Small', and 'Medium'.
     - `weight`: The weight of the fruit.
   - **Attributes**:
     - **Sizes**: The dataset categorizes fruit sizes into 'Tiny', 'Large', 'Small', and 'Medium'.
     - **Colors**: The dataset includes various colors such as 'Yellow', 'Pink', 'Pale Yellow', 'Red', 'Creamy White', 'Green', 'Purple', and 'Black'.
     - **Weight**: The weight of the fruit is summarized with the following statistics:
       - **Count**: 200
       - **Mean**: 59.05 grams
       - **Standard Deviation**: 46.70 grams
       - **Minimum**: 1 gram
       - **25th Percentile (Q1)**: 8.14 grams
       - **Median (50th Percentile)**: 63.11 grams
       - **75th Percentile (Q3)**: 94.37 grams
       - **Maximum**: 250 grams
   - **Unique Values**:
     - `fruit_type`: 3 unique types ('Banana', 'Apple', 'Grape')
     - `color`: 9 unique colors
     - `size`: 5 unique sizes
     - `weight`: 81 unique weights

2. **Data Preprocessing**
     - Dataset: The dataset contains various features of fruits such as size, color, and weight.
     - Initial Setup: Imported libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, and several `sklearn` modules.
     - Data Loading: The dataset was loaded using `pandas.read_excel`.
     - Data Cleaning: Unnecessary columns were dropped, and categories with similar values were merged (e.g., `Largee` with `Large` and `Yellow1` with `Yellow`).
     - Feature Encoding: Categorical data was encoded using `LabelEncoder`.
     - Train-Test Split: The dataset was split into training and testing sets using `train_test_split` in 80/20 proportion.

3. **Exploratory Data Analysis (EDA)**
     - Data Shape & Structure: Basic exploration to understand the size and structure of the dataset.
     - Missing Values: Checked for missing values to ensure data completeness.
     - Data Description: Statistical description of numerical data to understand distributions and central tendencies.

4. **Outliers Identification**
   - **Method**: Used a boxplot method to identify and remove outliers from the 'weight' column.
   - **Process**:
     - **Calculate Quartiles**: Determine the first quartile (Q1) and third quartile (Q3) for the 'weight' column. These quartiles divide the data into four equal parts.
     - **Compute Interquartile Range (IQR)**: The IQR is calculated as the difference between Q3 and Q1. It measures the range within which the central 50% of the data falls.
     - **Determine Bounds**: Calculate the lower and upper bounds for acceptable data values. Any data points below the lower bound or above the upper bound are considered outliers.
     - **Filter Data**: Create a filtered dataset that only includes values within the determined bounds, effectively removing the outliers.
     
     ```python
     Q1 = data['weight'].quantile(0.25)
     Q3 = data['weight'].quantile(0.75)
     IQR = Q3 - Q1

     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR

     data_filtered = data[(data['weight'] >= lower_bound) & (data['weight'] <= upper_bound)]
     ```
   - **Boxplot Visualization**: Below is a boxplot visualization used to identify outliers in the 'weight' column.
        ![Boxplot](https://github.com/azamatgalidenov/fruitsPrediction/blob/main/img/boxplot.png)

5. **Model Building**
   - Models Used:
     - **Decision Tree Classifier**: Implemented with a focus on optimizing depth and features.
     - **Logistic Regression**: Applied with a grid search for hyperparameter tuning.
   - Feature Interaction for Logistic Regression:
     - Implemented feature interaction of weight and color, which significantly improved the model's accuracy from 0.6700 to 0.7750.
   - Hyperparameter Tuning:
     - Used `GridSearchCV` to find the best parameters for each model.
     - **Decision Tree Classifier Hyperparameters**:
        - `max_depth`: `None`
        - `min_samples_split`: 5
        - `min_samples_leaf`: 2
        - `criterion`: `entropy`
        - `max_features`: `None`
        - `random_state`: 42
        - `class_weight`: `balanced`
        - `max_leaf_nodes`: `None`
        - `min_impurity_decrease`: 0.01
        - `splitter`: `best
     - **Logistic Regression Hyperparameters**:
        - `penalty`: `'l2'`
        - `C`: 0.1
        - `solver`: `'lbfgs'`
        - `max_iter`: 300
        - `class_weight`: `None`
        - `l1_ratio`: `0`
   - Evaluation Metrics: Models were evaluated based on accuracy, F1-score, ROC-AUC, and classification reports.
    - **Decision Tree Classifier Hyperparameters**:
        - Accuracy: 0.8000
        - F1 Score: 0.8013
        - ROC-AUC Score: 0.8858
    - **Logistic Regression Hyperparameters**:
        - Accuracy: 0.7750
        - F1 Score: 0.7760
        - ROC-AUC Score: 0.9004

6. **Results**
   - Best Models:
     - **Optimized Decision Tree**: Provided a robust model with the best accuracy and F1 score.
     - **Optimized Logistic Regression**: Showed competitive performance with balanced precision and recall. The implementation of feature interaction in Logistic Regression improved accuracy from 0.6700 to 0.7750.

## Results and Conclusions

The project successfully demonstrated the ability to predict fruit types based on their physical characteristics. The Decision Tree Classifier, after hyperparameter tuning, emerged as the most accurate model. The use of logistic regression also provided insights into the linear separability of the data, although it was slightly less accurate than the Decision Tree model.

The implementation of feature interaction in Logistic Regression significantly improved its accuracy, highlighting the impact of feature engineering on model performance.

The project emphasizes the importance of data preprocessing, feature engineering, and model selection in achieving accurate predictions. Further improvements could include exploring additional features or alternative machine learning models.

## How to Use

Run the `Fruits_prediction.ipynb` notebook in a Jupyter environment to explore the code and reproduce the results.