import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import seaborn as sns

def __main__():
    df = pd.read_csv('Titanic-Dataset.csv')
    print(df.head(), "\n")

    print(df.isnull().sum(), "\n")
    
    df = df.fillna({'Age' : df.Age.median()})
    df = df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis = 1).dropna()

    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

    print(df.head(20), "\n")

    print(df.describe(), "\n")

    df.drop(['Sex', 'Survived'], axis = 1).hist(figsize = (10, 8))
    plt.show()

    df.drop(['Sex', 'Survived'], axis = 1).boxplot(figsize = (10, 8))
    plt.show()

    df = df.loc[df['Fare'] <= 100]

    X = df.drop('Survived', axis = 1)
    y = df['Survived']

    plt.figure(figsize = (10, 6))
    sns.heatmap(df.corr(), annot = True)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTreeClassifier(max_depth = 3)
    model.fit(X_train, y_train)

    plt.figure(figsize = (10, 8))
    plot_tree(model, feature_names = list(df.drop('Survived', axis = 1).columns), filled = True, class_names = ['Not Survived', 'Survived'])
    plt.show()

    predictions = model.predict(X_test)

    print(f'Accuracy: {accuracy_score(predictions, y_test)}')
    print(f'Recall: {recall_score(predictions, y_test)}')
    print(f'Precision: {precision_score(predictions, y_test)}')
    print(f'F1: {f1_score(predictions, y_test)}')
    print(f'ROC-AUC: {roc_auc_score(predictions, y_test)}\n')

    print(classification_report(predictions, y_test))

    print(model.feature_importances_)

    confusion = confusion_matrix(predictions, y_test)
    plt.figure(figsize = (8, 6))
    sns.heatmap(confusion, annot = True)
    plt.xlabel('Predictions')
    plt.ylabel('Real values')
    plt.show()

    predictions_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, predictions_proba)

    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, label = 'ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()


if __name__ == '__main__':
    __main__()

