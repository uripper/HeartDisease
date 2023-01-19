import RandomForestModel
import LogisticRegressionModels
import KNearestNeighborsModel
import matplotlib.pyplot as plt
import sklearn

logistic_regression_metrics = LogisticRegressionModels.LogisticRegressionModels().run()
random_forest_metrics = RandomForestModel.RandomForestModel().run()
k_nearest_neighbors_metrics = KNearestNeighborsModel.KNearestNeighborsModel().run()
