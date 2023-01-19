import pandas as pd

class Pipeline():
    def __init__(self, dataset):
        self.dataset_in_memory = pd.read_csv(dataset)
        self.medical_term_dictionary = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholestoral in mg/dl', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting Electrocardiographic Results', 'thalach': 'Maximum Heart Rate Achieved', 'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression Induced by Exercise Relative to Rest', 'slope': 'Slope of the Peak Exercise ST Segment', 'ca': 'Number of Major Vessels (0-3) Colored by Flourosopy', 'thal': 'Thalium Stress Test Result', 'target': 'Heart Disease Diagnosis'}
    def process(self):
        self.dependent_variable = self.dataset_in_memory['target']
        self.independent_variables = self.dataset_in_memory.drop('target', axis=1)
        return self.dataset_in_memory, self.dependent_variable, self.independent_variables, self.medical_term_dictionary
    def shuffle(self):
        shuffled_dataset_in_memory = self.dataset_in_memory.sample(frac=1).reset_index(drop=True)
        shuffled_dependent_variable = shuffled_dataset_in_memory['target']
        shuffled_independent_variables = shuffled_dataset_in_memory.drop('target', axis=1)
        return shuffled_dataset_in_memory, shuffled_dependent_variable, shuffled_independent_variables
        
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
    