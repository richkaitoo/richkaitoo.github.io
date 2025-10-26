---
layout: single
title: "Evaluating Predictive Model Performance for Cardio Health Services Demand: A Comparative Analysis"
excerpt: "This study compares various machine learning classification models that have the capability to predict heart disease."
date: 2025-01-03
categories:
  - projects
tags:
  - data
  - scientist
  - science
  - analyst
  - Python
classes: wide
header:
  overlay_image: /assets/images/heart.jpeg  
  overlay_filter: 0.3  # darkens the image for better text contrast (0 = no filter, 1 = black)
  caption: ""  # leave empty if you don’t want a caption
  show_overlay_excerpt: false
  image_description: "Heart disease prediction research"
---



# Problem Overview

According to CDC, Heart disease is the leading cause of death for people of most racial and ethnic groups in the United States. Reports shows that about 805,000 people in the United States have a heart attack. Among these huge numbers, CDC states that 605,000  are first heart attack cases, and rest of the 200,000 happen to be people who have already had a heart attack. Looking at these numbers, available data is being modellled such that the case is being detected before it happens, it will save alot of lives and the presure on facilities that serves these people will reduces drastically. In view this, this research uses available data to model heart attack. 


##  Research Question
In order to identify the most effective machine learning model for heart disease prediction a short list of machine learning models will be chosen and their performance be compared to each.The following research questions will guide this project.
- Q1. What is the best performing model among the chosen once?
- Q2. Among the models selected, does each of their perfoamce change after hyperparameter tuning?
- Q3. How does using k-fold cross-validation provide a more robust estimate of model performance compared to a single train-test split?

### Full Code: [Github](https://github.com/ernselito/Heart-Attack-Risk-Prediction/blob/main/Heart_Disease_Prediction.ipynb)

##  Dataset
The dataset that will be used for this project is fetched from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). According to the publisher on Kaggle, the dataset consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. The structure of the data is summarized below.

The data is loaded for the Kaggle API using the following

```python
print('Data source import complete.')
data_path = kagglehub.dataset_download('johnsmith88/heart-disease-dataset')
heart = pd.read_csv(data_path + '/heart.csv')
heart.head()
```
ge	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
0	52	1	0	125	212	0	1	168	0	1.0	2	2	3	0
1	53	1	0	140	203	1	0	155	1	3.1	0	0	3	0
2	70	1	0	145	174	0	1	125	1	2.6	0	0	3	0
3	61	1	0	148	203	0	1	161	0	0.0	2	1	3	0
4	62	0	0	138	294	1	1	106	0	1.9	1	3	2	0

The above shows the first five rows of the dataset. 

# Exploratoty Analysis
The data structure was checked, missingness checked and follwoing summary is provided.   
- **Size**: 1,025 records with 14 features
- **No Missing Values**: Complete dataset with no null values

### Features Description and Target
This stuyd will use the following variables as the feature variables.
- **Demographic**: age, sex
- **Medical History**: cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol)
- **Test Results**: fbs (fasting blood sugar), restecg (resting electrocardiographic results)
- **Exercise-related**: thalach (maximum heart rate), exang (exercise induced angina), oldpeak (ST depression)
- **Other Medical**: slope, ca (number of major vessels), thal (thalassemia)
The target variable for this study is chosen below.
- **Target Variable**: Binary classification (0 = no disease, 1 = disease)

#### Examining relationships between variables

The visual relationship between the numerical variables is explored. The heatmaps examining the relationships between numerical variables is shown below.
```python
corr_matrix = heart.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
```

![Correlatio Map](/assets/images/correlation_matrix.png)

The output suggest the following:

- *Strong positive correlations*: `cp` and `target` (0.43), `thalach` and `slope` (0.40), `exang` and `oldpeak` (-0.44), `target` and `thalach` (0.42)
- *Strong negative correlations*: `target` and `sex` (-0.28), `exang` and `cp` (-0.40), `target` and `ca` (-0.38)

We will move on to the methodology that will be used in answering the research question.

#  Methodology
The section answers the step by step methods that was implemented to answer the research questions.
First of all, to avoid data leaking, the data spliting will take place before all other transformations are applied to the data. 
```python
X_train, X_test, y_train, y_test = train_test_split(heart.drop('target', axis=1), heart['target'], test_size=0.2, random_state=42)
```
The dataset is splitted into 80-20 train test set. 

### 1. Data Preprocessing
- **Data Splitting**: 75-25 train-test split
- **Feature Engineering**:
The function below process the data into:
  - Age binning into 6 groups
  - One-hot encoding for categorical variables
  - Standard scaling for numerical features

```python
def preprocess_data(X_train, X_test):
    categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_columns = ['trestbps', 'chol', 'thalach', 'oldpeak']

    # Define preprocessing steps
    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_preprocessor = StandardScaler()
    age_preprocessor = KBinsDiscretizer(n_bins=6, encode='onehot-dense', strategy='uniform')

    # Use ColumnTransformer to apply different preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_preprocessor, numerical_columns),
            ('age', age_preprocessor, ['age']),
            ('cat', categorical_preprocessor, categorical_variables)
        ]
    )

    # Fit and transform the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get the feature names
    feature_names = []
    feature_names += [f"{col}" for col in numerical_columns]
    feature_names += [f"age_{i+1}" for i in range(6)]
    for var in categorical_variables:
        if var == 'age':
            continue
        categories = X_train[var].unique()
        feature_names += [f"{var}_{cat}" for cat in categories]

    return X_train_preprocessed, X_test_preprocessed, feature_names

# Usage
X_train_preprocessed, X_test_preprocessed, feature_names = preprocess_data(X_train, X_test)

# Convert the preprocessed data to DataFrames with feature names
X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)
```
  

### 2. Modelling
Since the target is a binary  (0 = no disease, 1 = disease), this suggest that it is a classification model. There are so many classification model available but the follwing were selected due to because they are the most common models. Apart from that, they have shown to perform best in cases where the target is binary.
- Logistic Regression
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Support Vector Machines (SVMs)
- Gradient Boosting

The following represent how the models are set up. 

```python
def define_model():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors (KNN, k=3)": KNeighborsClassifier(n_neighbors=3),
        "Decision Trees": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machines (SVMs)": SVC(),
        "K-Nearest Neighbors (KNN, default k)": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    return models

def train_and_evaluate_models(X_train_df, y_train, X_test_df, y_test, models = define_model()):
    # Train and evaluate the models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    return results

results = train_and_evaluate_models(X_train_df, y_train, X_test_df, y_test, models = define_model())
for name, accuracy in results.items():
    print(f"Model: {name}, Accuracy: {accuracy:.3f}")
```
Model: Logistic Regression, Accuracy: 0.795
Model: Linear Discriminant Analysis (LDA), Accuracy: 0.820
Model: K-Nearest Neighbors (KNN, k=3), Accuracy: 0.902
Model: Decision Trees, Accuracy: 0.985
Model: Random Forest, Accuracy: 0.985
Model: Support Vector Machines (SVMs), Accuracy: 0.683
Model: K-Nearest Neighbors (KNN, default k), Accuracy: 0.732
Model: Gradient Boosting, Accuracy: 0.932

From the above output, it was found that, the models can be categorized as:

- Top Models:
    - Decision Trees: 98.5%
    - Random Forest: 98.5%
    - Gradient Boosting: 93.2%
- Middle Models:
    - K-Nearest Neighbors (KNN, k=3): 90.2%
    - LDA: 82%
    - Logistic Regression: 79.5%
- Lower Models:
    - K-Nearest Neighbors (KNN, default k): 73.2%
    - Support Vector Machines (SVMs): 68.3%


### 3. Model Selection Process
The section marks the process of crossvalidating and fine tuning the model. We start of by cross validating and the next section will be the hyperparametr tuning.

```python
def train_and_evaluate_models(X_train_df, y_train, X_test_df, y_test, models = define_model()):
    
    # Train and evaluate the models with cross-validation
    results = {}
    cv_results = {}
    for name, model in models.items():
        model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)
        results[name] = accuracy_score(y_test, y_pred)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_df, y_train, cv=5, scoring='accuracy')
        cv_results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'accuracy_scores': cv_scores
        }

    return results, cv_results

# Usage
results, cv_results = train_and_evaluate_models(X_train_df, y_train, X_test_df, y_test, models = define_model())

# Print the results
for name, accuracy in results.items():
    print(f"Model: {name}, Accuracy: {accuracy:.3f}")
    print(f"  Cross-Validation Mean Accuracy: {cv_results[name]['mean_accuracy']:.3f}")
    print(f"  Cross-Validation Std Accuracy: {cv_results[name]['std_accuracy']:.3f}")
    print()

```
The results indicated the top tier models to be:

1. Random Forest: Accuracy: 0.985, Cross-Validation Mean Accuracy: 0.982
2. Decision Trees: Accuracy: 0.971, Cross-Validation Mean Accuracy: 0.976
3. K-Nearest Neighbors (KNN, k=3): Accuracy: 0.951, Cross-Validation Mean Accuracy: 0.909

These three models stand out from the rest due to their high accuracy and cross-validation mean accuracy scores. They demonstrate strong performance and stability across different folds of the data.

Hence, these three models will be selected for further hyperparameter tuning.

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grids for top 3 models
param_grids = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    }
}

# Perform grid search for each model
best_models = {}

for name, model_info in param_grids.items():
    grid_search = GridSearchCV(
        model_info["model"],
        model_info["params"],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_df, y_train)

    best_models[name] = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_
    }

    print(f"Model: {name}")
    print("Best Parameters:", grid_search.best_params_)
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print()

```
The results of the out is shown below: 

Model: Random Forest
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Best Score: 0.9841

Model: Decision Tree
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best Score: 0.9793

Model: K-Nearest Neighbors
Best Parameters: {'n_neighbors': 3, 'p': 2, 'weights': 'distance'}
Best Score: 0.9890

After using the GridSearch to hyperparamter tune the Random Forest Mode, it was found out that, the best perform model is KNN with k=3, p=2. Comparing this to the best score of Random Forest, achieved by the model earlier, this is very close to the original score (0.9817). This indicate that the model is KNN has performed well, and the hyperparameter tuning did result in a significant improvement.

### Hyperparameter Tuning Results
- **K-Nearest Neighbors**: Best Score - 98.9%
  - Optimal parameters: n_neighbors=3, p=2, weights='distance'
- **Random Forest**: Best Score - 98.4%
- **Decision Trees**: Best Score - 97.9%

### Final Model Performance
After hyperparameter tuning, the optimized KNN model achieved:
- **Test Accuracy**: 100%
- **Confusion Matrix**: Perfect classification (102 true negatives, 103 true positives)
- **Classification Report**: All metrics (precision, recall, F1-score) at 1.00

##  Key Findings

1. **Top Performing Models**: Random Forest, Decision Trees, and KNN demonstrated the highest accuracy
2. **Best Model**: K-Nearest Neighbors with optimized parameters achieved perfect prediction
3. **Feature Importance**: Strong correlations observed between target and features like chest pain type (cp), maximum heart rate (thalach), and exercise-induced angina (exang)

##  Insights & Recommendations

- **Clinical Application**: The high accuracy suggests potential for clinical decision support systems
- **Model Robustness**: Cross-validation confirmed model stability across different data splits
- **Feature Importance**: Chest pain type and exercise test results are strong predictors
- **Future Work**: Explore deep learning approaches and additional clinical features

##  Conclusion
This project successfully demonstrates that machine learning models, particularly K-Nearest Neighbors with proper parameter tuning, can achieve exceptional performance in heart disease prediction. The results highlight the potential of ML in medical diagnostics and provide a robust framework for similar healthcare prediction tasks.

