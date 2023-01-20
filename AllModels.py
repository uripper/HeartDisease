import RandomForestModel
import LogisticRegressionModels
import KNearestNeighbor
import Adaboost
import Xgboost
import matplotlib.pyplot as plt
import sklearn

logistic_regression_metrics = LogisticRegressionModels.LogisticRegressionModels().run()
random_forest_metrics = RandomForestModel.RandomForestModel().run()
k_nearest_neighbors_metrics = KNearestNeighbor.KNearestNeighbor().run()
adaboost_metrics = Adaboost.Adaboost().run()
xgboost_metrics = Xgboost.Xgboost().run()

accuracy_all = [logistic_regression_metrics[0], random_forest_metrics[0], k_nearest_neighbors_metrics[0], adaboost_metrics[0], xgboost_metrics[0]]
cr_all = [logistic_regression_metrics[2], random_forest_metrics[2], k_nearest_neighbors_metrics[2], adaboost_metrics[2], xgboost_metrics[2]]
roc_all = [logistic_regression_metrics[3], random_forest_metrics[3], k_nearest_neighbors_metrics[3], adaboost_metrics[3], xgboost_metrics[3]]
roc_auc_all = [logistic_regression_metrics[4], random_forest_metrics[4], k_nearest_neighbors_metrics[4], adaboost_metrics[4], xgboost_metrics[4]]
prec_rec_all = [logistic_regression_metrics[5], random_forest_metrics[5], k_nearest_neighbors_metrics[5], adaboost_metrics[5], xgboost_metrics[5]]
all_precisions = []
all_recalls = []
for dictionary in cr_all:
    for key, value in dictionary.items():
        if key != "macro avg":
            pass
        else:
            for new_key, new_value in value.items():
                if new_key == "precision":
                    all_precisions.append(new_value)
                elif new_key == "recall":
                    all_recalls.append(new_value)
                
                
    
# Plotting accuracy of all the models
plot = plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.3)
plt.bar(["Logistic Regression", "Random Forest", "K Nearest Neighbors", "Adaboost", "Xgboost"], accuracy_all)
plt.title("Accuracy of all the models")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.ylim(.8, 1)
# put the number of each bar above the bar
plt.text(0, round(accuracy_all[0], 4), round(accuracy_all[0], 4), ha='center', va='bottom')
plt.text(1, round(accuracy_all[1], 4), round(accuracy_all[1], 4), ha='center', va='bottom')
plt.text(2, round(accuracy_all[2], 4), round(accuracy_all[2], 4), ha='center', va='bottom')
plt.text(3, round(accuracy_all[3], 4), round(accuracy_all[3], 4), ha='center', va='bottom')
plt.text(4, round(accuracy_all[4], 4), round(accuracy_all[4], 4), ha='center', va='bottom')
ax.set_xticklabels(["Logistic Regression", "Random Forest", "K Nearest Neighbors", "Adaboost", "Xgboost"], rotation=45)
plt.savefig("AccuracyAllModels.png")

plt.close()

#Plotting the precision
plot = plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.3)
plt.bar(["Logistic Regression", "Random Forest", "K Nearest Neighbors", "Adaboost", "Xgboost"], all_precisions)
plt.title("Precision of all the models")
plt.xlabel("Models")
plt.ylabel("Precision")

plt.ylim(.8, 1)
plt.text(0, round(all_precisions[0], 4), round(all_precisions[0], 4), ha='center', va='bottom')
plt.text(1, round(all_precisions[1], 4), round(all_precisions[1], 4), ha='center', va='bottom')
plt.text(2, round(all_precisions[2], 4), round(all_precisions[2], 4), ha='center', va='bottom')
plt.text(3, round(all_precisions[3], 4), round(all_precisions[3], 4), ha='center', va='bottom')
plt.text(4, round(all_precisions[4], 4), round(all_precisions[4], 4), ha='center', va='bottom')
ax.set_xticklabels(["Logistic Regression", "Random Forest", "K Nearest Neighbors", "Adaboost", "Xgboost"], rotation=45)
plt.savefig("PrecisionAllModels.png")
plt.close()

# Plotting the recall
plot = plt.figure()
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.3)
plt.bar(["Logistic Regression", "Random Forest", "K Nearest Neighbors", "Adaboost", "Xgboost"], all_recalls)
plt.title("Recall of all the models")
plt.xlabel("Models")
plt.ylabel("Recall")

plt.ylim(.8, 1)
plt.text(0, round(all_recalls[0], 4), round(all_recalls[0], 4), ha='center', va='bottom')
plt.text(1, round(all_recalls[1], 4), round(all_recalls[1], 4), ha='center', va='bottom')
plt.text(2, round(all_recalls[2], 4), round(all_recalls[2], 4), ha='center', va='bottom')
plt.text(3, round(all_recalls[3], 4), round(all_recalls[3], 4), ha='center', va='bottom')
plt.text(4, round(all_recalls[4], 4), round(all_recalls[4], 4), ha='center', va='bottom')
ax.set_xticklabels(["Logistic Regression", "Random Forest", "K Nearest Neighbors", "Adaboost", "Xgboost"], rotation=45)
plt.savefig("RecallAllModels.png")
plt.close()