import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
class Pipeline():
    def __init__(self, dataset):
        self.dataset_in_memory = pd.read_csv(dataset)
        self.medical_term_dictionary = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholestoral in mg/dl', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting Electrocardiographic Results', 'thalach': 'Maximum Heart Rate Achieved', 'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression Induced by Exercise Relative to Rest', 'slope': 'Slope of the Peak Exercise ST Segment', 'ca': 'Number of Major Vessels (0-3) Colored by Flourosopy', 'thal': 'Thalium Stress Test Result', 'target': 'Heart Disease Diagnosis'}
        self.test_size = .2
    def process(self):
        self.dataset_in_memory = self.dataset_in_memory.sample(frac=1)
        self.dataset_in_memory.rename(columns=self.medical_term_dictionary, inplace=True)
        return self.dataset_in_memory
    def accuracy(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    def train_test_split(self, dataset=None):
        if dataset is None:
            dataset = self.dataset_in_memory
        test, train = np.split(dataset, [int(self.test_size*len(dataset))])
        test_dependent_variable = test['Heart Disease Diagnosis']
        test_independent_variables = test.drop('Heart Disease Diagnosis', axis=1)
        train_dependent_variable = train['Heart Disease Diagnosis']
        train_independent_variables = train.drop('Heart Disease Diagnosis', axis=1)
        return test_dependent_variable, test_independent_variables, train_dependent_variable, train_independent_variables
    
    
     
if __name__ == '__main__':
    pipeline = Pipeline('Dataset\heart.csv')
    dataset, dependent_variable, independent_variables, medical_term_dictionary = pipeline.process()
    shuffled_dataset, shuffled_dependent_variable, shuffled_independent_variables = pipeline.shuffle()
    print("Medical Term Dictionary: ")
    print("Column Name:    Medical Term")
    for key, value in medical_term_dictionary.items():
        print(f"{key}:     {value}")
    print("Dataset Unshuffled: ")
    print(dataset.head())
    input("Continue? ")
    print("Dependent Variable unshuffled: ")
    print(dependent_variable.head())
    input("Continue? ")
    print("Independent Variables unshuffled: ")
    print(independent_variables.head())
    input("Continue? ")
    print("Dataset Shuffled: ")
    print(shuffled_dataset.head())
    input("Continue? ")
    print("Dependent Variable Shuffled: ")
    print(shuffled_dependent_variable.head())
    input("Continue? ")
    print("Independent Variables Shuffled: ")
    print(shuffled_independent_variables.head())
    