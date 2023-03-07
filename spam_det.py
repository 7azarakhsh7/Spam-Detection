import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from  sklearn.model_selection import cross_val_score
data = pd.read_table("SMSSpamCollection", header = None, names = ["labels", "texts"])

raw_labels = data["labels"]
raw_features = data.drop("labels", axis = 1)
print(raw_features.shape)

label_dict = {'spam': 1, 'ham': 0}
encoded_labels  = raw_labels.apply(lambda x: label_dict[x])

count_vectorizer = CountVectorizer()
vectorized_features = count_vectorizer.fit_transform(raw_features["texts"])

print(vectorized_features.shape)

X_train,X_test, y_train, y_test = train_test_split(vectorized_features, encoded_labels, test_size=0.2, shuffle=True, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = BernoulliNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
precision = precision_score(y_true=y_test, y_pred=y_pred)
recall = recall_score(y_true=y_test, y_pred=y_pred)
f_score = fbeta_score(y_true=y_test, y_pred=y_pred, beta = 0.5)
print("precision: {:.3f}".format(precision))
print("recall: {:.3f}".format(recall))
print("f_score: {:.3f}".format(f_score))

scores = cross_val_score(clf, vectorized_features, encoded_labels, cv=7, scoring='precision')

mean = scores.mean()
std = scores.std()
print("precision: {:.3f} +- {:.3f}".format(mean, 2*std))

