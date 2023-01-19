
import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sklearn


class RandomForestModel():
    def __init__(self):
        pass

    def run(self):
        pipeline = Pipeline.Pipeline('Dataset//heart.csv')
        data = pipeline.process()
        test_dependent_variable, test_independent_variables, train_dependent_variable, train_independent_variables = pipeline.train_test_split()

        # Create a Random Forest Classifier
        RandomForestClassifierModel = RandomForestClassifier(n_estimators=100)
        RandomForestClassifierModel.fit(train_independent_variables, train_dependent_variable)
        predictions = RandomForestClassifierModel.predict(test_independent_variables)
        probability = RandomForestClassifierModel.predict_proba(test_independent_variables)
        accuracy, confusion_matrix, classification_report, roc_curve, roc_auc, precision_recall_curve = pipeline.metrics(predictions, test_dependent_variable, "RandomForest")
        return accuracy, confusion_matrix, classification_report, roc_curve, roc_auc, precision_recall_curve
if __name__ == '__main__':
    rf_dictionary = RandomForestModel().run()






