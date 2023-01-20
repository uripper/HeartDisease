import Pipeline
from sklearn.ensemble import AdaBoostClassifier

class Adaboost():
    def __init__(self):
        pass

    def run(self):
        pipeline = Pipeline.Pipeline('Dataset//heart.csv')
        data = pipeline.process()
        test_dependent_variable, test_independent_variables, train_dependent_variable, train_independent_variables = pipeline.train_test_split()
        AdaboostClassifierModel = AdaBoostClassifier(n_estimators=100)
        AdaboostClassifierModel.fit(train_independent_variables, train_dependent_variable)
        predictions = AdaboostClassifierModel.predict(test_independent_variables)
        probability = AdaboostClassifierModel.predict_proba(test_independent_variables)
        accuracy, confusion_matrix, classification_report, roc_curve, roc_auc, precision_recall_curve = pipeline.metrics(predictions, test_dependent_variable, "Adaboost")
        return accuracy, confusion_matrix, classification_report, roc_curve, roc_auc, precision_recall_curve
    
    
if __name__ == '__main__':
    metrics = Adaboost().run()