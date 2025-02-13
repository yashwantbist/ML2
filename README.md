# ML2
The lab focuses on handling missing values, training an initial machine learning model, and evaluating its performance.

**Part 1: Before We Start (Handling Missing Values)
**Before training the model, we need to ensure that our data is clean and free of missing values. This step involves:
![2](https://github.com/user-attachments/assets/6ce40d99-d7bb-4d62-897d-d0f46f6ddbb5)


Identifying and handling missing values in the dataset.
Avoiding premature judgments based on data values to ensure unbiased predictions.

Part 2: Training and Evaluating an Initial Model
The next step is to train a model to determine the relationship between features and the target variable. The procedure includes:
![3](https://github.com/user-attachments/assets/7e748902-e181-422c-9e2e-5fa0a168bd2f)


Separating Features and Target Columns:

X_train = df_num.drop('price', axis=1)
y_train = df_num['price']
Creating an Appropriate Model with Suitable Hyper-Parameters: Using a Random Forest Regressor:

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
Part 3: Finding R-Squared
To measure how well the model fits the training data, we use the R-squared metric:

r2 = rf.score(X_train, y_train)
print(f"{r2:.4f}")

![4](https://github.com/user-attachments/assets/a20be15f-d09c-467d-9f60-8e58680d12f7)

Part 4: Commenting on R-Squared Value
A perfect R-squared score is 1.0, indicating the model perfectly recalls the training data.
An R-squared score of 0 means the model performs no better than always returning the average price.
A high R-squared score suggests a possible relationship between features and the target, captured by the model.
If the R-squared score is low, it indicates no relationship or the model's inability to capture it.

How to Run
Clone the repository:
git clone <repository_url>
Navigate to the project directory:
cd <project_directory>
Open the Jupyter Notebook:
jupyter notebook
Run the notebook cells to execute the code.
