import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay
import numpy as np
from matplotlib import pyplot as plt
import random
class Pipeline():
    def __init__(self, dataset):
        self.dataset_in_memory = pd.read_csv(dataset)
        self.medical_term_dictionary = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type', 'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholestoral in mg/dl', 'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting Electrocardiographic Results', 'thalach': 'Maximum Heart Rate Achieved', 'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression Induced by Exercise Relative to Rest', 'slope': 'Slope of the Peak Exercise ST Segment', 'ca': 'Number of Major Vessels (0-3) Colored by Flourosopy', 'thal': 'Thalium Stress Test Result', 'target': 'Heart Disease Diagnosis'}
        self.test_size = .4
        self.colors = ['rosybrown', 'lightcoral', 'indianred', 'firebrick', 'darkred', 'red', 'mistyrose','salmon', 'darksalmon', 'lightsalmon', 'crimson', 'red', 'orangered', 'tomato', 'coral', 'darkorange', 'orange', 'gold', 'yellow', 'lightyellow', 'lemonchiffon', 'lightgoldenrodyellow', 'papayawhip', 'moccasin', 'peachpuff', 'palegoldenrod', 'khaki', 'darkkhaki', 'olive', 'yellowgreen', 'darkolivegreen', 'olivedrab', 'lawngreen', 'chartreuse', 'greenyellow', 'darkseagreen', 'mediumspringgreen', 'springgreen', 'mediumseagreen', 'seagreen', 'forestgreen', 'green', 'darkgreen', 'yellowgreen', 'limegreen', 'lime', 'lawngreen', 'lightgreen', 'palegreen', 'darkseagreen', 'mediumspringgreen', 'springgreen', 'mediumseagreen', 'seagreen', 'forestgreen', 'green', 'darkgreen', 'aqua', 'cyan', 'lightcyan', 'paleturquoise', 'aquamarine', 'turquoise', 'mediumturquoise', 'darkturquoise', 'lightseagreen', 'cadetblue', 'darkcyan', 'teal', 'powderblue', 'lightblue', 'lightskyblue', 'skyblue', 'deepskyblue', 'lightsteelblue', 'dodgerblue', 'cornflowerblue', 'steelblue', 'royalblue', 'blue', 'mediumblue', 'darkblue', 'navy', 'midnightblue', 'lavender', 'thistle', 'plum', 'violet', 'orchid', 'fuchsia', 'magenta', 'mediumorchid', 'mediumpurple', 'blueviolet', 'darkviolet', 'darkorchid', 'darkmagenta', 'purple', 'indigo', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mintcream', 'azure', 'aliceblue', 'seashell', 'beige', 'oldlace', 'lavenderblush', 'mistyrose', 'gainsboro', 'lightgray', 'silver', 'darkgray', 'gray', 'dimgray', 'lightslategray']

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
        elif name == "Report":
            not_heart_disease = data["0"]
            heart_disease = data["1"]
            accuracy = data["accuracy"]
            macro_avg = data["macro avg"]
            weighted_avg = data["weighted avg"]
            current_color = random.choice(self.colors)
            print(current_color)

            # Precision

            plot = plt.figure()
            plt.bar(['Not Heart Disease', 'Heart Disease', 'Macro Avg', 'Weighted Average'], [not_heart_disease["precision"], heart_disease["precision"], macro_avg["precision"], weighted_avg["precision"]], color=current_color)
            plt.title('Precision')
            minimum = min([not_heart_disease["precision"], heart_disease["precision"], macro_avg["precision"], weighted_avg["precision"]])
            maximum = max([not_heart_disease["precision"], heart_disease["precision"], macro_avg["precision"], weighted_avg["precision"]])
            if maximum > 1 - 0.1:
                plt.ylim(minimum-.1, 1)
            else:
                plt.ylim(minimum-.1, maximum+.1)
            plt.savefig(f'Graphs/{directory}//Precision.png')
            plt.close()

            # Recall

            plot = plt.figure()
            plt.bar(['Not Heart Disease', 'Heart Disease', 'Macro Avg', 'Weighted Average'], [not_heart_disease["recall"], heart_disease["recall"], macro_avg["recall"], weighted_avg["recall"]], color=current_color)
            plt.title('Recall')
            minimum = min([not_heart_disease["recall"], heart_disease["recall"], macro_avg["recall"], weighted_avg["recall"]])
            maximum = max([not_heart_disease["recall"], heart_disease["recall"], macro_avg["recall"], weighted_avg["recall"]])
            if maximum > 1 - 0.1:
                plt.ylim(minimum-.1, 1)
            else:
                plt.ylim(minimum-.1, maximum+.1)
            plt.savefig(f'Graphs/{directory}//Recall.png')
            plt.close()

            # F1 Score
            plot = plt.figure()
            plt.bar(['Not Heart Disease', 'Heart Disease', 'Macro Avg', 'Weighted Average'], [not_heart_disease["f1-score"], heart_disease["f1-score"], macro_avg["f1-score"], weighted_avg["f1-score"]], color=current_color)
            plt.title('F1 Score')
            minimum = min([not_heart_disease["f1-score"], heart_disease["f1-score"], macro_avg["f1-score"], weighted_avg["f1-score"]])
            maximum = max([not_heart_disease["f1-score"], heart_disease["f1-score"], macro_avg["f1-score"], weighted_avg["f1-score"]])
            if maximum > 1 - 0.1:
                plt.ylim(minimum-.1, 1)
            else:
                plt.ylim(minimum-.1, maximum+.1)
            plt.savefig(f'Graphs/{directory}//F1Score.png')
            plt.close()

            # Support
            plot = plt.figure()
            plt.bar(['Not Heart Disease', 'Heart Disease'], [not_heart_disease["support"], heart_disease["support"]], color=current_color)
            plt.title('Support')
            minimum = min([not_heart_disease["support"], heart_disease["support"]])
            maximum = max([not_heart_disease["support"], heart_disease["support"]])
            plt.ylim(minimum-10, maximum+10)
            plt.savefig(f'Graphs/{directory}//Support.png')
            plt.close()
            
    
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
            self.graphing('Report', name, working_cr_score)
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
        return accuracy_score(y_test, y_pred)
    def auc_score(self, y_test, y_pred):
        return auc(y_test, y_pred)
    def confusion_matrix_score(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)
    def classification_report_score(self, y_test, y_pred):
        return classification_report(y_test, y_pred, output_dict=True)
    def roc_curve_score(self, y_test, y_pred):
        return roc_curve(y_test, y_pred)
    def roc_auc(self, y_test, y_pred):
        return roc_auc_score(y_test, y_pred)
    def precision_recall_curve_score(self, y_test, y_pred):
        return precision_recall_curve(y_test, y_pred)
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
    