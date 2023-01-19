
import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModels():
    def __init__(self):
        pass
    def run(self):
        pipeline = Pipeline.Pipeline('Dataset//heart.csv')
        dataset = pipeline.process()
        test_dependent_variable, test_independent_variables, train_dependent_variable, train_independent_variables = pipeline.train_test_split(dataset=dataset)


        # Creating the Linear Regression Models, using the different solvers
        LogisticRegressionModel_lbfgs = LogisticRegression(solver='lbfgs', max_iter=1000)
        LogisticRegressionModel_liblinear = LogisticRegression(solver='liblinear', max_iter=1000)
        LogisticRegressionModel_lbfgs.fit(train_independent_variables, train_dependent_variable)
        LogisticRegressionModel_liblinear.fit(train_independent_variables, train_dependent_variable)
        LogisticRegressionModel_newton_cg = LogisticRegression(solver='newton-cg', max_iter=10000)
        LogisticRegressionModel_newton_cg.fit(train_independent_variables, train_dependent_variable)
        LogisticRegressionModel_sag = LogisticRegression(solver='sag', max_iter=10000)
        LogisticRegressionModel_sag.fit(train_independent_variables, train_dependent_variable)
        LogisticRegressionModel_saga = LogisticRegression(solver='saga', max_iter=10000)
        LogisticRegressionModel_saga.fit(train_independent_variables, train_dependent_variable)
        LogisticRegressionModel_newton_cholesky = LogisticRegression(solver='newton-cholesky', max_iter=10000)
        LogisticRegressionModel_newton_cholesky.fit(train_independent_variables, train_dependent_variable)

        # Predicting the test set results
        predictions_lbfgs = LogisticRegressionModel_lbfgs.predict(test_independent_variables)
        predictions_liblinear = LogisticRegressionModel_liblinear.predict(test_independent_variables)
        predictions_newton_cg = LogisticRegressionModel_newton_cg.predict(test_independent_variables)
        predictions_sag = LogisticRegressionModel_sag.predict(test_independent_variables)
        predictions_saga = LogisticRegressionModel_saga.predict(test_independent_variables)
        predictions_newton_cholesky = LogisticRegressionModel_newton_cholesky.predict(test_independent_variables)
        predictions = [predictions_lbfgs, predictions_liblinear, predictions_newton_cg, predictions_sag, predictions_saga, predictions_newton_cholesky]
        predictions_list = ['lbfgs', 'liblinear', 'newton cg', 'sag', 'saga', 'newton cholesky']
        accuracy_dict = {}
        for prediction_index in range(len(predictions)):
            name = predictions_list[prediction_index]
            result = predictions[prediction_index]
            accuracy_dict[name] = pipeline.accuracy(result, test_dependent_variable)
            
        average = sum(accuracy_dict.values())/len(accuracy_dict)
        accuracy_dict['average'] = average
        return accuracy_dict
    
if __name__ == '__main__':
    accuracy_dict = LogisticRegressionModels().run()
    print(accuracy_dict)
    plot = plt.figure(figsize=(10, 5))
    # plt.xticks(rotation=45)
    plt.title('Logistic Regression Models')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.bar(range(len(accuracy_dict)), list(accuracy_dict.values()), align='center')
    plt.ylim(.8, .9)
    plt.xticks(range(len(accuracy_dict)), list(accuracy_dict.keys()))
    #put the value of the bar on top of the bar
    for i, v in enumerate(list(accuracy_dict.values())):
        v= round(v, 4)
        string_v = str(v)
        before_decimal = string_v.replace('.', '')[1:3]
        after_decimal = string_v.replace('.', '')[3:]
        string_v = before_decimal + '.' + after_decimal + '%'
        plt.text(i-.1, v+.001, string_v)
    

    plot.savefig('LogisticRegressionModels.png')