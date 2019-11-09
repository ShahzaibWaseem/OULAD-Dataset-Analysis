import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

# fetches the data from .csv files
def fetchData(filename):
	return pd.read_csv("../Dataset/" + filename, header=0)

# replaces a particular string from a particular column in a particular DataFrame
def removeUnwantedData(inputDataFrame, columnName, removeString, replaceString):
	inputDataFrame[columnName] = inputDataFrame[columnName].str.replace(removeString, replaceString)
	return inputDataFrame

# convert NaNs to some predefined data
def resolveNANs(inputDataFrame):
	for columnName in inputDataFrame.columns:
		if inputDataFrame[columnName].dtype == "object":
			inputDataFrame[columnName] = inputDataFrame[columnName].fillna("Nothing")
		elif inputDataFrame[columnName].dtype == "int64":
			inputDataFrame[columnName] = inputDataFrame[columnName].fillna(0)
		elif inputDataFrame[columnName].dtype == "float64":
			inputDataFrame[columnName] = inputDataFrame[columnName].fillna(0.0)
	return inputDataFrame

# Removing unwanted data from the columns
def encodingColumns(inputDataFrame):
	inputDataFrame=removeUnwantedData(inputDataFrame, "disability", "N", "1")
	inputDataFrame=removeUnwantedData(inputDataFrame, "disability", "Y", "2")

	inputDataFrame=removeUnwantedData(inputDataFrame, "final_result", "Fail", "1")
	inputDataFrame=removeUnwantedData(inputDataFrame, "final_result", "Withdrawn", "2")
	inputDataFrame=removeUnwantedData(inputDataFrame, "final_result", "Pass", "3")
	inputDataFrame=removeUnwantedData(inputDataFrame, "final_result", "Distinction", "4")

	inputDataFrame=removeUnwantedData(inputDataFrame, "highest_education", "No Formal quals", "1")
	inputDataFrame=removeUnwantedData(inputDataFrame, "highest_education", "Lower Than A Level", "2")
	inputDataFrame=removeUnwantedData(inputDataFrame, "highest_education", "A Level or Equivalent", "3")
	inputDataFrame=removeUnwantedData(inputDataFrame, "highest_education", "HE Qualification", "4")
	inputDataFrame=removeUnwantedData(inputDataFrame, "highest_education", "Post Graduate Qualification", "5")
	return inputDataFrame

# Plots the Confusion Matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()