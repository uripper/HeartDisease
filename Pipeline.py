import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay
import numpy as np
class Pipeline():
    def __init__(self, dataset):
        self.dataset_in_memory = pd.read_csv(dataset)
        self.medical_term_dictionary = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholestoral in mg/dl', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting Electrocardiographic Results', 'thalach': 'Maximum Heart Rate Achieved', 'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression Induced by Exercise Relative to Rest', 'slope': 'Slope of the Peak Exercise ST Segment', 'ca': 'Number of Major Vessels (0-3) Colored by Flourosopy', 'thal': 'Thalium Stress Test Result', 'target': 'Heart Disease Diagnosis'}
        self.test_size = .5
    def process(self):
        self.dataset_in_memory = self.dataset_in_memory.sample(frac=1)
        self.dataset_in_memory.rename(columns=self.medical_term_dictionary, inplace=True)
        return self.dataset_in_memory
    
    def graphing(self, name, directory, data):
        if name == 'ConfusionMatrix':
            cm_graph = ConfusionMatrixDisplay(confusion_matrix=data, display_labels=['Not Heart Disease','Heart Disease']).plot()
            cm_graph.figure_.savefig(f'Graphs/{directory}//ConfusionMatrix.png')
        elif name == 'ROC':
            roc_graph = RocCurveDisplay(fpr=data[0], tpr=data[1]).plot()
            roc_graph.figure_.savefig(f'Graphs/{directory}//ROC.png')
    
    def metrics(self, y_test, y_pred, name):
        try:
            working_accuracy = self.accuracy(y_test, y_pred)
        except ValueError:
            # print("Cannot calculate accuracy score")
            working_accuracy = "N/A"
        try:
            working_auc = self.auc_score(y_test, y_pred)
        except ValueError:
            # print("Cannot calculate auc score")
            working_auc = "N/A"
        try:
            working_cm_score = self.confusion_matrix_score(y_test, y_pred)
            self.graphing('ConfusionMatrix', name, working_cm_score)
        except ValueError:
            # print("Cannot calculate confusion matrix score")
            working_cm_score = "N/A"
        try:
            working_cr_score = self.classification_report_score(y_test, y_pred)
        except ValueError:
            # print("Cannot calculate classification report score")
            working_cr_score = "N/A"
        try:
            working_roc_curve_score = self.roc_curve_score(y_test, y_pred)
            self.graphing('ROC', name, working_roc_curve_score)

        except ValueError:
            # print("Cannot calculate roc curve score")
            working_roc_curve_score = "N/A"
        try:
            working_roc_auc = self.roc_auc(y_test, y_pred)
        except ValueError:
            # print("Cannot calculate roc auc score")
            working_roc_auc = "N/A"
        try:
            working_precision_recall_curve_score = self.precision_recall_curve_score(y_test, y_pred)
        except ValueError:
            # print("Cannot calculate precision recall curve score")
            working_precision_recall_curve_score = "N/A"
        
        return working_accuracy, working_cm_score, working_cr_score, working_roc_curve_score, working_roc_auc, working_precision_recall_curve_score
    
    def accuracy(self, y_test, y_pred):
        working_accuracy = accuracy_score(y_test, y_pred)
        return working_accuracy
    def auc_score(self, y_test, y_pred):
        working_auc = auc(y_test, y_pred)
        return working_auc
    def confusion_matrix_score(self, y_test, y_pred):
        working_cm_score = confusion_matrix(y_test, y_pred)
        return working_cm_score
    def classification_report_score(self, y_test, y_pred):
        working_cr_score = classification_report(y_test, y_pred)
        return working_cr_score
    def roc_curve_score(self, y_test, y_pred):
        working_roc_curve_score = roc_curve(y_test, y_pred)
        return working_roc_curve_score
    def roc_auc(self, y_test, y_pred):
        working_roc_auc = roc_auc_score(y_test, y_pred)
        return working_roc_auc
    def precision_recall_curve_score(self, y_test, y_pred):
        working_precision_recall_curve_score = precision_recall_curve(y_test, y_pred)
        return working_precision_recall_curve_score
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
    