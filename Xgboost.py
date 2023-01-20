import Pipeline
from xgboost import XGBClassifier

class Xgboost():
    def __init__(self):
        pass
    def run(self):
        pipeline = Pipeline.Pipeline('Dataset//heart.csv')
        dataset = pipeline.process()
        test_dependent_variable, test_independent_variables, train_dependent_variable, train_independent_variables = pipeline.train_test_split(dataset=dataset)
        xgboost = XGBClassifier()
        xgboost.fit(train_independent_variables, train_dependent_variable)
        predictions = xgboost.predict(test_independent_variables)
        predictions_proba = xgboost.predict_proba(test_independent_variables)
        accuracy, confusion_matrix, classification_report, roc_curve, roc_auc, precision_recall_curve = pipeline.metrics(predictions, test_dependent_variable, "Xgboost")
        return accuracy, confusion_matrix, classification_report, roc_curve, roc_auc, precision_recall_curve
    
    
if __name__ == '__main__':
    Xgboost().run()