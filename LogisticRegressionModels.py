
import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import sklearn

class LogisticRegressionModels():
    def __init__(self):
        pass
    def run(self):
        pipeline = Pipeline.Pipeline('Dataset//heart.csv')
        dataset = pipeline.process()
        test_dependent_variable, test_independent_variables, train_dependent_variable, train_independent_variables = pipeline.train_test_split(dataset=dataset)


        # Creating the Linear Regression Models, using the different solvers
        LogisticRegressionModel_lbfgs = LogisticRegression(solver='lbfgs', max_iter=1000)
        LogisticRegressionModel_lbfgs.fit(train_independent_variables, train_dependent_variable)
        
        # Predicting the test set results
        predictions_lbfgs = LogisticRegressionModel_lbfgs.predict(test_independent_variables)
        prob_predictions_lbfgs = LogisticRegressionModel_lbfgs.predict_proba(test_independent_variables)
  
        accuracy, cm, cr, roc, roc_auc, prec_rec = pipeline.metrics(predictions_lbfgs, test_dependent_variable, "LogisticRegression")
    
            
        
        return accuracy, cm, cr, roc, roc_auc, prec_rec
    
if __name__ == '__main__':
    accuracy_dict = LogisticRegressionModels().run()
    