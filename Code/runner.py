import utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC

def main():
	requiredStudentInfo=["id_student", "highest_education", "studied_credits", "num_of_prev_attempts", "final_result", "disability"]
	requiredStudentAssessment=["id_student", "date_submitted", "score"]
	requiredStudentVLE=["id_student", "sum_of_sum_click"]
	# reading csv Files
	studentInfo = utils.fetchData("studentInfo.csv")
	studentAssessment = utils.fetchData("studentAssessment.csv")
	studentVLE = utils.fetchData("studentVle.csv")

	# creating a new column sum_of_sum_click
	studentVLE["sum_of_sum_click"] = studentVLE.groupby(["id_student"])["sum_click"].transform(sum)

	studentInfo.set_index('id_student')
	studentAssessment.set_index('id_student')
	studentVLE.set_index('id_student')

	studentInfo=studentInfo[requiredStudentInfo]
	studentAssessment=studentAssessment[requiredStudentAssessment]
	studentVLE=studentVLE[requiredStudentVLE]
	studentVLE.drop_duplicates("id_student", inplace=True)

	# Theres are some "?" in studentAssessment csv
	# replacing them with 0 and converting to integer
	print("Cleaning \"Score\" Column in studentAssessment.csv")
	studentAssessment = utils.removeUnwantedData(studentAssessment, "score", "?", "0")
	studentAssessment["score"] = pd.to_numeric(studentAssessment["score"])

	# combining the three dataFrames
	print("Combining dataFrames...")
	combinedDF = studentInfo.combine_first(studentAssessment)
	combinedDF = combinedDF.combine_first(studentVLE)

	combinedDF.set_index('id_student')
	combinedDFcopy=combinedDF.copy()

	# converting string based data to dummy columns
	print("Encoding string columns...")
	combinedDF = utils.encodingColumns(combinedDF)

	combinedDF["disability"] = pd.to_numeric(combinedDF["disability"])
	combinedDF["final_result"] = pd.to_numeric(combinedDF["final_result"])
	combinedDF["highest_education"] = pd.to_numeric(combinedDF["highest_education"])

	# resolving NAN which are created when we combined the dataFrames
	print("Resolving NANs...")
	combinedDF = utils.resolveNANs(combinedDF)

	# Applying KMeans Clustering to create a new column in the dataFrame "procastinate"
	print("Applying KMeans...")
	kmeans = KMeans(init='random', n_clusters=2,  tol=1e-04, random_state=0).fit(combinedDF[["highest_education", "studied_credits", "num_of_prev_attempts", "final_result", "disability", "date_submitted", "score", "sum_of_sum_click"]])
	# labels = kmeans.fit_predict(combinedDF)
	labels=kmeans.labels_

	# changing 1's and 0's to True and False
	combinedDFcopy["procastinate"] = labels == 1

	# Adding a new column "procastinate"
	combinedDF["procastinate"] = labels

	# Randomizing
	combinedDF.sample(frac=1)

	# Creating New DataFrames inTime and procastinate (for Visualizatino)
	inTime, procastinate = [x for _, x in combinedDF.groupby(combinedDF['procastinate'] == 0)]
	inTime=inTime.head(100)
	procastinate=procastinate.head(100)
	# print(procastinate)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(procastinate["date_submitted"], procastinate["score"], procastinate["final_result"], c="r", marker="o")
	ax.scatter(inTime["date_submitted"], inTime["score"], inTime["final_result"], c="g", marker="o")

	ax.set_xlabel("date_submitted")
	ax.set_ylabel("score")
	ax.set_zlabel("final_result")
	plt.title("Scatter Plot")
	plt.show()

	# Exporting the dataFrame to csv
	export_csv = combinedDF.to_csv ('../Dataset/studentFinal.csv', index=False, header=True)

	# Setting X and y
	y=combinedDF["procastinate"]
	X=combinedDF.drop("procastinate", axis=1)
	X.set_index('id_student', inplace=True)

	# Splitting Data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)

	# ANN
	print("Running ANN...")
	ann = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
	ann.fit(X_train, y_train.values.ravel())
	predictions = ann.predict(X_test)
	score = ann.score(X_test, y_test)
	loss_values = ann.loss_curve_
	print("Accuracy: " , score * 100)
	plt.title("ANN Loss")
	plt.ylabel("Loss Value")
	plt.plot(loss_values)
	plt.show()

	utils.plot_confusion_matrix(confusion_matrix(y_test,predictions), ["procastinate", "in Time"])

	print("Classification Report\n", classification_report(y_test,predictions))

	# Logistic Regression
	print("Running Logistic Regression...")
	logisticRegr = LogisticRegression()
	logisticRegr.fit(X_train, y_train)
	predictions = logisticRegr.predict(X_test)
	score = logisticRegr.score(X_test, y_test)
	print("Accuracy: ", score * 100)

	utils.plot_confusion_matrix(confusion_matrix(y_test,predictions), ["procastinate", "in Time"])
	print("Classification Report\n", classification_report(y_test,predictions))

	# SVM
	print("Running SVM...")
	svmClassifier=SVC(kernel='linear')
	svmClassifier.fit(X_train.head(1000), y_train[:1000])
	predictions = svmClassifier.predict(X_test)
	score = svmClassifier.score(X_test, y_test)
	print("Accuracy: ", score * 100)

	utils.plot_confusion_matrix(confusion_matrix(y_test,predictions), ["procastinate", "in Time"])
	print("Classification Report\n", classification_report(y_test,predictions))

if __name__ == '__main__':
	main()