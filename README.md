
## Dataset Information

**Dataset Name:** Iris
**Source: UCI Machine Learning Repository**
**Number of Rows:** 150  
**Number of Columns:** 5
**Exploratory Data Analysis and Summary Statistics**

Summary statistics of the dataset:

| Column Name   | Data Type |
|---------------|-----------|
| sepal_length  | float64   |
| sepal_width   | float64   |
| petal_length  | float64   |
| petal_width   | float64   |
| class         | object    |

```python
# Summary statistics
print(data.describe())
```
| Statistic  | Sepal Length | Sepal Width | Petal Length | Petal Width |
|------------|--------------|-------------|--------------|-------------|
| Count      | 150          | 150         | 150          | 150         |
| Mean       | 5.843        | 3.054       | 3.759        | 1.199       |
| Std Dev    | 0.828        | 0.434       | 1.764        | 0.763       |
| Min        | 4.300        | 2.000       | 1.000        | 0.100       |
| 25%        | 5.100        | 2.800       | 1.600        | 0.300       |
| 50% (Median)| 5.800        | 3.000       | 4.350        | 1.300       |
| 75%        | 6.400        | 3.300       | 5.100        | 1.800       |
| Max        | 7.900        | 4.400       | 6.900        | 2.500       |

```python
sns.boxplot(x=data['sepal_length'])
plt.title('Sepal Length Box Plot')
plt.show()
```
```python
# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
```
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split features and target variable
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
```

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Iris-setosa      | 1.00      | 1.00   | 1.00     | 10      |
| Iris-versicolor  | 1.00      | 0.92   | 0.96     | 12      |
| Iris-virginica   | 0.92      | 1.00   | 0.96     | 8       |
| **Accuracy**     |           |        | 0.97     | 30      |
| **Macro Avg**    | 0.97      | 0.97   | 0.97     | 30      |
| **Weighted Avg** | 0.97      | 0.97   | 0.97     | 30      |


**Results and Findings**

In this project, significant findings have been obtained by analyzing the Iris dataset. The RandomForestClassifier model was used to classify flower species with high accuracy.

***Future Work***

Similar analyses can be conducted with additional datasets.
Comparison of different machine learning models to find the best performer.
Feature engineering to improve results.
