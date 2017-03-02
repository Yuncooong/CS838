from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def evaluation(y_hat, y, verbose=True):
	TP_idx = set(np.where(y == 1)[0]) & set(np.where(y_hat == 1)[0])
	precision = len(TP_idx) / sum(y_hat == 1)
	recall = len(TP_idx) / sum(y == 1)
	f1 = 2 * precision * recall / (precision + recall)
	if verbose:
		print "precision:", precision
		print "recall:", recall
		print "F1:", f1
	return precision, recall, f1

def getErrors(y_hat, y, title):
	fp = y_hat[:].astype(bool) & ~y[:].astype(bool)
	fn = ~y_hat[:].astype(bool) & y[:].astype(bool)
	result = {"FP":[], "FN":[]}
	for i in range(0, len(y)):
		if fp[i]:
			filename, word = fp.index[i]
			num = int(filename.split(".")[0])
			result["FP"].append([num, word])
		elif fn[i]:
			filename, word = fp.index[i]
			num = int(filename.split(".")[0])
			result["FN"].append([num, word])
	result["FP"] = sorted(result["FP"])
	result["FN"] = sorted(result["FN"])
	with open(title, "w+") as f:
		f.write("false positives\n")
		for x in result["FP"]:
			x[0] = ".".join([str(x[0]), "txt"])
			f.write(",".join(x) + "\n")
		f.write("false negatives\n")
		for x in result["FN"]:
			x[0] = ".".join([str(x[0]), "txt"])
			f.write(",".join(x) + "\n")


if __name__ == '__main__':
	# load the data
	I = pd.DataFrame.from_csv("./trainset.csv", index_col=[0, 1])
	J = pd.DataFrame.from_csv("./testset.csv", index_col=[0, 1])
	feature_names = list(I)[:-1]
	label_name = list(I)[-1]

	result_ols = [] #[[sensitivity, specificity], ...]
	result_lr = []
	result_dt = []
	result_rf = []
	result_svm = []
	kf = KFold(n_splits = 10, shuffle=True)  # 10 fold CV
	for train_idx, test_idx in kf.split(range(0, len(I))):
		X_train = I[feature_names].iloc[train_idx]
		y_train = I[label_name].iloc[train_idx]
		X_test = I[feature_names].iloc[test_idx]
		y_test = I[label_name].iloc[test_idx]

		# linear regression
		model = LinearRegression()
		model.fit(X_train, y_train)
		y_hat = model.predict(X_test)
		y_hat = y_hat > 0.5
		y_hat = y_hat.astype(int)
		precision, recall, f1 = evaluation(y_hat, y_test, False)
		result_ols.append([precision, recall, f1])

		# logistic regression
		model = LogisticRegression()
		model.fit(X_train, y_train)
		y_hat = model.predict_proba(X_test)
		y_hat = y_hat[:,1] > 0.5
		y_hat = y_hat.astype(int)
		precision, recall, f1 = evaluation(y_hat, y_test, False)
		result_lr.append([precision, recall, f1])

		# decision tree
		model = DecisionTreeClassifier(criterion="entropy", class_weight = {1:0.25, 0:0.75})
		model.fit(X_train, y_train)
		y_hat = model.predict(X_test)
		precision, recall, f1 = evaluation(y_hat, y_test, False)
		result_dt.append([precision, recall, f1])

		# random forest
		model = RandomForestClassifier(criterion="entropy", class_weight = {1:0.25, 0:0.75})
		model.fit(X_train, y_train)
		y_hat = model.predict(X_test)
		precision, recall, f1= evaluation(y_hat, y_test, False)
		result_rf.append([precision, recall, f1])

		# svm
		model = svm.SVC()#class_weight = {1:0.4, 0:0.6})
		model.fit(X_train, y_train)
		y_hat = model.predict(X_test)
		precision, recall, f1 = evaluation(y_hat, y_test, False)
		result_svm.append([precision, recall, f1])

	ols_precision, ols_recall, ols_f1 = np.array(result_ols).mean(0)  # mean by column
	lr_precision, lr_recall, lr_f1 = np.array(result_lr).mean(0)
	dt_precision, dt_recall, dt_f1 = np.array(result_dt).mean(0)
	rf_precision, rf_recall, rf_f1 = np.array(result_rf).mean(0)
	svm_precision, svm_recall, svm_f1 = np.array(result_svm).mean(0)

	print "----- 10-fold CV on training set -----"
	print "ordinay least square regression:"
	print "precision:", ols_precision, "\trecall:", ols_recall, "\tf1:", ols_f1
	print "logistic regression:"
	print "precision:", lr_precision, "\trecall:", lr_recall, "\tf1:", lr_f1
	print "decision tree:"
	print "precision:", dt_precision, "\trecall:", dt_recall, "\tf1:", dt_f1
	print "random forest:"
	print "precision:", rf_precision, "\trecall:", rf_recall, "\tf1:", rf_f1
	print "svm:"
	print "precision:", svm_precision, "\trecall:", svm_recall, "\tf1:", svm_f1


