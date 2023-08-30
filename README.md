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



