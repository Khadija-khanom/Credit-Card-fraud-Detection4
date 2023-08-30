# Comparative Analysis of Machine Learning and Deep Learning Algorithms for Credit Card Fraud Detection on 4th Dataset

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/3e7c53bd-c3a2-406a-ba9a-1c1d3f2c2fc5)


**Dataset Description:** The dataset contains information related to credit applications or loan approval. It consists of 25,134 entries (individual applicants) and 20 columns (features). Here's a breakdown of the columns and their meanings:

**Unnamed:** 0: An index or identifier for the rows.

**ID:** A unique identifier for each individual applicant.

**GENDER:** The gender of the applicant (e.g., 'M' for male, 'F' for female).

**CAR:** Whether the applicant owns a car ('Y' for yes, 'N' for no).

**REALITY:** Whether the applicant owns real estate ('Y' for yes, 'N' for no).

**NO_OF_CHILD:** The number of children the applicant has.

**INCOME:** The income of the applicant.

**INCOME_TYPE:** The type of income source (e.g., 'Working', 'Commercial associate', etc.).

**EDUCATION_TYPE:** The type of education the applicant has (e.g., 'Secondary / secondary special').

**FAMILY_TYPE:** The type of family the applicant belongs to (e.g., 'Married', 'Single / not married').

**HOUSE_TYPE:** The type of housing the applicant lives in (e.g., 'House / apartment').

**FLAG_MOBIL:** A binary flag indicating if the applicant has a mobile phone (1 for yes, 0 for no).

**WORK_PHONE:** A binary flag indicating if the applicant has a work phone (1 for yes, 0 for no).

**PHONE:** A binary flag indicating if the applicant has a phone (1 for yes, 0 for no).

**E_MAIL:** A binary flag indicating if the applicant has an email (1 for yes, 0 for no).

**FAMILY_SIZE:** The size of the applicant's family.

**BEGIN_MONTH:** The month when the applicant started the loan application process.

**AGE:** The age of the applicant.

**YEARS_EMPLOYED:** The number of years the applicant has been employed.

**TARGET:** The target variable indicating loan approval status (0 for approved, 1 for not approved).

**Data Types:**

The dataset includes features of different data types:

- Object data type (e.g., GENDER, CAR, INCOME_TYPE) represents categorical variables.
- Float64 data type (e.g., INCOME, FAMILY_SIZE) represents numerical variables with decimal values.
- Int64 data type (e.g., ID, NO_OF_CHILD, BEGIN_MONTH) represents integer variables.

**Summary:** This dataset appears to be suitable for building models to predict loan approval based on various applicant attributes. It includes information such as personal characteristics (gender, age), financial status (income, assets), employment details (years employed), and family-related information (family size, marital status). The target variable "TARGET" indicates whether an applicant's loan was approved or not.

Table of Contents
=================


##Data Visualization

**Calculate the correlation matrix**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/7503aa62-ea7d-4eb8-be37-e353cc92a87d)

**Distribution of selected features**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/f42d8dda-2d96-4fa7-be1b-e33733b3de48)

**Box plot of age grouped by loan approval status**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/72fef34c-b9b1-4c74-a06f-ecabeb31ba2f)

**A pair plot to visualize relationships between numerical variables**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/8331ac8d-a647-4f22-a7b4-d2998dde80c7)

**Box plot of income grouped by gender and loan approval status**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/37b8799a-4b62-4176-899a-59e85af65740)

**Create age groups and plot distribution**
![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/476ab611-5424-47da-9c39-d7e153d3348d)

**Create a pair plot after data processing to visualize relationships between numerical variables (one-hot encoded)**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/71df6c9c-1c78-4197-a7c7-5b4a4bad0bfd)

**Ratio of Frauds and Non- Frauds data**
![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/f77286a2-5d26-47d1-8625-0336eb97a660)

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/233c9127-8904-43fa-b8bf-b9ee03492b47)

**Distribution after data processing**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/32126ea1-8f72-4f01-9941-5566274ab8e7)

# Implementation process Of deep learning models

## Convolutional Neural Network (CNN) Model 

Presenting the overview of the main steps in CNN model building process:

**1. Data Preparation:**

- Separate target variable (TARGET) from features (X).
- Perform one-hot encoding on categorical features.
- Convert the target variable to binary labels.
- Apply SMOTE for balancing the data.
- Split the balanced data into training and test sets.
- Perform feature scaling using StandardScaler.
- Reshape data for CNN input format.

**2. CNN Model Architecture:**

- Define a function (create_cnn_model()) to create the CNN architecture.
- Include a 1D convolutional layer, max pooling, flattening, dense layers, and dropout.
- Compile the model with binary cross-entropy loss and the Adam optimizer.

**3. Hyperparameter Tuning:**

- Wrap the Keras model with KerasClassifier.
- Define hyperparameters for tuning, such as filters, kernel sizes, dropout rates, epochs, and batch sizes.
- Create a GridSearchCV object (grid_cnn) with the wrapped model and hyperparameters.

**4. Model Training and Tuning:**

- Fit the GridSearchCV object on the training data.
- Find the best CNN model using grid_cnn.best_estimator_.
  
**5. Final Model Training:**

- Train the best CNN model on the training data.

**6. Prediction and Evaluation:**

- Make predictions on the test data using the trained best CNN model.
- Calculate continuous predictions and binary predictions by thresholding.
- Evaluate model performance using accuracy, classification report, and confusion matrix.
  
In summary, this process involves data preprocessing, CNN model architecture definition, hyperparameter tuning with GridSearchCV, training the model, making predictions, and evaluating the model's performance on the test data.


## Recurrent Neural Network (RNN) Model

The overview of the main steps in building an RNN model for binary classification is given below:

**1. Data Preparation:**

- Reshape the scaled data for RNN input format.
- Convert the training and test sets to the required shape of (number_of_samples, 1, number_of_features).

**2. RNN Model Architecture:**

- Define a function (create_rnn_model()) to create the RNN architecture.
- Include an LSTM layer followed by dropout for regularization.
- Compile the model with binary cross-entropy loss and the Adam optimizer.

**3. Hyperparameter Tuning:**

- Wrap the Keras RNN model with KerasClassifier.
- Define hyperparameters for tuning, such as units (LSTM units), dropout rates, epochs, and batch sizes.
- Create a GridSearchCV object (grid_rnn) with the wrapped model and hyperparameters.
  
**4. Model Training and Tuning:**

- Fit the GridSearchCV object on the training data.
- Find the best RNN model using grid_rnn.best_estimator_.
  
**5. Final Model Training:**

Train the best RNN model on the training data.

**6. Prediction and Evaluation:**

- Make predictions on the test data using the trained best RNN model.
- Calculate continuous predictions and binary predictions by thresholding.
- Evaluate model performance using accuracy, classification report, and confusion matrix.
  
In summary, this process involves data reshaping, RNN model architecture definition, hyperparameter tuning using GridSearchCV, training the model, making predictions, and evaluating the model's performance on the test data.

# Evaluating the performance of deep learning models

## Convolutional Neural Network (CNN) Model 

The summary of the results is presented in a table, followed by a description of each metric:

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/9409c8d2-6a35-436b-a595-e54e34945a76)

**Description:**

   **1. Best CNN Accuracy:** The accuracy of the best CNN model on the test data is approximately 98.46%. This indicates the proportion of correctly predicted instances among all instances in the test set.

   **2. Precision (Class 0):** Precision for class 0 (loan not approved) is 98.0%. This means that out of all instances the model predicted as class 0, 98% were truly class 0.

   **3. Precision (Class 1):** Precision for class 1 (loan approved) is 99.0%. This means that out of all instances the model predicted as class 1, 99% were truly class 1.

   **4. Recall (Class 0):** Recall (also known as sensitivity or true positive rate) for class 0 is 99.0%. This indicates the proportion of actual class 0 instances that were correctly predicted by the model.

  **5. Recall (Class 1):** Recall for class 1 is 98.0%. This indicates the proportion of actual class 1 instances that were correctly predicted by the model.

  **6. F1-Score (Class 0):** The F1-score for class 0 is 0.985. The F1-score is the harmonic mean of precision and recall and provides a balanced measure of the two metrics.

  **7. F1-Score (Class 1):** The F1-score for class 1 is 0.985. This is the harmonic mean of precision and recall for class 1.

  **8. Support (Class 0):** The number of instances belonging to class 0 in the test set is 4894.

  **9. Support (Class 1):** The number of instances belonging to class 1 in the test set is 4991.

The results suggest that the CNN model has performed well on the test data with high accuracy and balanced precision and recall values for both classes. The model's performance is consistent across both classes, indicating its effectiveness in predicting loan approvals.


## Recurrent Neural Network (RNN) Model

The summary of the RNN model's results is presented in a table, followed by a description of each metric:

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/6e678b99-98f0-4de3-9164-4700267ce963)

**Description:**

   **1. Best RNN Accuracy:** The accuracy of the best RNN model on the test data is approximately 98.95%. This indicates the proportion of correctly predicted instances among all instances in the test set.

   **2. Precision (Class 0):** Precision for class 0 (loan not approved) is 99.0%. This means that out of all instances the model predicted as class 0, 99% were truly class 0.

   **3. Precision (Class 1):** Precision for class 1 (loan approved) is 99.0%. This means that out of all instances the model predicted as class 1, 99% were truly class 1.

   **4. Recall (Class 0):** Recall (sensitivity) for class 0 is 99.0%. This indicates the proportion of actual class 0 instances that were correctly predicted by the model.

   **5. Recall (Class 1):** Recall for class 1 is 99.0%. This indicates the proportion of actual class 1 instances that were correctly predicted by the model.

   **6. F1-Score (Class 0):** The F1-score for class 0 is 0.990. The F1-score is the harmonic mean of precision and recall and provides a balanced measure of the two metrics.

   **7. F1-Score (Class 1):** The F1-score for class 1 is 0.990. This is the harmonic mean of precision and recall for class 1.

   **8. Support (Class 0):** The number of instances belonging to class 0 in the test set is 4894.

   **9. Support (Class 1):** The number of instances belonging to class 1 in the test set is 4991.

The results suggest that the RNN model has performed exceptionally well on the test data with high accuracy and balanced precision and recall values for both classes. The model's performance is consistent across both classes, indicating its effectiveness in predicting loan approvals.

## Learning curve of CNN and RNN model


**Learning Curve of CNN model**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/93901538-68c4-4f66-bbc8-b50ca70fd5d2)

**Learning Curve of RNN model**

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/a6732330-358a-4dac-b555-502eafb1e57f)

# Implementation process Of Machine learning models 

## Model Building Process

demonstrating the process of building and evaluating several machine learning models for binary classification on a dataset. Here's an overall explanation of the model building process:

**Data Preparation:**

- The dataset is loaded into a DataFrame, and the target variable 'TARGET' is separated from the feature variables.
- Categorical features are identified for one-hot encoding.
- One-hot encoding is applied to the categorical features, converting them into numerical format.
- The target variable is transformed into binary labels using a LabelEncoder.
- SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset by oversampling the minority class.
  
**Data Splitting and Scaling:**

- The balanced dataset is split into training and test sets using the train_test_spli function.
- Feature scaling is performed using StandardScaler to ensure that all features have the same scale.

### Decision Tree Classifier

- Hyperparameters are defined for Decision Tree model tuning, including criteria, max depth, min samples split, and min samples leaf.
- A Decision Tree model is created.
- GridSearchCV is used to perform hyperparameter tuning using cross-validation.
- The best Decision Tree model is obtained from GridSearchCV.
- The best model is trained on the scaled training data, and predictions are made on the test data.
- Accuracy, classification report, and confusion matrix for the model are displayed.
  
### Random Forest Classifier

- Hyperparameters are defined for Random Forest model tuning, including the number of estimators, criteria, max depth, min samples split, and min samples leaf.
- A Random Forest model is created.
- GridSearchCV is used for hyperparameter tuning.
- The best Random Forest model is obtained from GridSearchCV.
- The best model is trained on the scaled training data, and predictions are made on the test data.
- Accuracy, classification report, and confusion matrix for the model are displayed.
  
### Logistic Regression

- Hyperparameters are defined for Logistic Regression model tuning, including regularization parameter (C), penalty, and solver.
- A Logistic Regression model is created.
- GridSearchCV is used for hyperparameter tuning.
- The best Logistic Regression model is obtained from GridSearchCV.
- The best model is trained on the scaled training data, and predictions are made on the test data.
- Accuracy, classification report, and confusion matrix for the model are displayed.
  
### K-Nearest Neighbors (KNN) Classifier

- Hyperparameters are defined for KNN model tuning, including the number of neighbors, weights, and distance metric (p).
- A KNN model is created.
- GridSearchCV is used for hyperparameter tuning.
- The best KNN model is obtained from GridSearchCV.
- The best model is trained on the scaled training data, and predictions are made on the test data.
- Accuracy, classification report, and confusion matrix for the model are displayed.
  
### Learning Curve Visualization

- Learning curves for each model are plotted using the plot_learning_curve function.
- Learning curves show how the models' performance changes as the training data size increases.

Overall, the code demonstrates the process of preparing the data, building multiple classification models with different algorithms (Decision Tree, Random Forest, Logistic Regression, KNN), tuning hyperparameters using GridSearchCV, training the best models, making predictions, and evaluating model performance. The learning curves visualize how the models' performance improves with more training data

# Evaluating the performance of machine learning models

Here are the results of the different models presented in a table format:

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/c97cf2ad-5037-4347-8349-56414655e187)


## Model Performance Analysis

- The Random Forest model achieved the highest accuracy of 0.996, followed closely by the Decision Tree model with an accuracy of 0.990.
- Both Random Forest and Decision Tree models have excellent precision, recall, and F1-scores, indicating strong performance across all metrics.
- Logistic Regression achieved a moderate accuracy of 0.908. While the precision and recall are balanced, it falls behind the other models in terms of accuracy.
- K-Nearest Neighbors (KNN) model achieved an accuracy of 0.992, showing strong performance similar to the Decision Tree model.

## Best Performing Model

- The Random Forest model performed the best in terms of accuracy, achieving 99.6%. It also displayed perfect precision, recall, and F1-scores, suggesting that it made very few errors in both positive and negative predictions.
- The Random Forest model's ensemble nature, which combines multiple decision trees, allows it to capture complex relationships in the data while mitigating overfitting.
- The model's ability to handle a variety of data distributions and features makes it well-suited for this dataset, resulting in excellent performance.
  
In summary, the Random Forest model is the best-performing model in this scenario due to its high accuracy, balanced precision and recall, and strong overall performance on the given dataset.

## Learning Curve of machine learning models

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/cff79aab-8688-40bf-ac41-b71059dc0a2e)

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/27231db1-5e74-45ba-bb4a-7be6b18f667c)

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/9ddad6d1-b843-4c75-b71e-920d06ad1dc7)

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/c005a730-b2f5-4955-8a92-88aece1b545e)

# Comparative analysis between machine learning and deep learning models

A comparative analysis of the model performances is presented below:

![image](https://github.com/Khadija-khanom/Credit-Card-fraud-Detection4/assets/138976722/0887acf2-bb13-4cb1-ba52-9be8d3cb0220)


**Comparative Analysis:**

- The Random Forest model achieves the highest accuracy of 0.9963, closely followed by the K-Nearest Neighbors (KNN) model with an accuracy of 0.9923.
- The Random Forest and KNN models have perfect precision, recall, and F1-scores for both classes, indicating near-perfect prediction capabilities.
- The Decision Tree and RNN models also exhibit strong performance with high accuracy, precision, recall, and F1-scores for both classes.
- The CNN model has a slightly lower accuracy of 0.9846, but it maintains high precision, recall, and F1-scores for both classes as well.
- The Logistic Regression model has the lowest accuracy of 0.9080, and while it has balanced precision and recall, it falls behind the other models in terms of accuracy and overall performance.

**Best Performing Model:** **The Random Forest model** stands out as the best performer in this scenario:

- It achieves the highest accuracy, precision, recall, and F1-scores across both classes.
- Its ensemble nature helps it capture complex relationships in the data, and it demonstrates robustness against overfitting.
- Random Forest is capable of handling a variety of data distributions and feature sets, which contributes to its excellent performance on this dataset.
  
In conclusion, the Random Forest model is the clear winner due to its exceptional performance on all metrics and its capability to handle the complexities of the given dataset effectively.




