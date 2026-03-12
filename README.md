# OIBSIP_DataAnalytics-_task2-L2
This project predicts wine quality based on chemical attributes such as acidity and density. Machine learning models including Random Forest, Stochastic Gradient Descent, and Support Vector Classifier are used to analyze the dataset and classify wine quality while visualizing patterns using data analysis libraries. 
# Wine Quality Prediction

## Objective
The objective of this project is to predict the quality of wine using machine learning algorithms based on its chemical properties. The project demonstrates how data analysis and classification models can be used to evaluate wine quality.

## Dataset
Dataset Link: (Add dataset link here)

The dataset contains various chemical attributes of wine such as acidity, density, and other properties that influence wine quality.

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## Steps Performed
1. Collected the wine quality dataset containing chemical properties of wine.
2. Explored the dataset to understand its structure and features.
3. Cleaned and prepared the dataset for analysis.
4. Analyzed chemical attributes such as acidity and density.
5. Implemented classification models including Random Forest, SGD, and Support Vector Classifier.
6. Trained the models and evaluated their performance.
7. Visualized patterns and insights using data visualization tools.

## Code (Example)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
Key Insights

Chemical properties influence wine quality.

Machine learning models can classify wine quality effectively.

Visualization helps understand feature relationships.

Outcome

The project successfully predicts wine quality using machine learning models and demonstrates how chemical characteristics influence classification results.

Author

Ayesha Asna
