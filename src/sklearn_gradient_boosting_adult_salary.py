# author by lifulong
# first version
# auc = 0.8684
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

train_file="./../data/adult.data"
test_file="./../data/adult.test"

# feature_list
feature_list=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
feature_type = ["continuous", "discreate", "continuous", "discreate", "continuous", "discreate", "discreate", "discreate", "discreate", "discreate", "continuous", "continuous", "continuous", "discreate", "label"]

train_dataset=pd.read_csv(train_file, sep=',', header=None, names=feature_list).values
test_dataset=pd.read_csv(test_file, sep=',', header=None, names=feature_list).values

labelEncoder=LabelEncoder()

discreate_features = [ index for (index,typ) in enumerate(feature_type) if typ == 'discreate' ]
label_features = discreate_features[:]
label_features.append(14)

for label in label_features:
	label_feature = np.vstack((train_dataset[:,label].reshape(-1,1), test_dataset[:,label].reshape(-1,1)))
	labelEncoder.fit(label_feature)
	train_dataset[:,label]=labelEncoder.transform(train_dataset[:,label])
	test_dataset[:,label]=labelEncoder.transform(test_dataset[:,label])

y_train = train_dataset[:,14].astype('int')
y_test = test_dataset[:,14].astype('int')
x_train = train_dataset[:, 0:13]
x_test = test_dataset[:, 0:13]

gradientBoostingClassifier=GradientBoostingClassifier()

gradientBoostingClassifier.fit(x_train, y_train)
#dump model to file
#joblib.dump(decisionTreeClassifier, 'decisionTree.pkl')

predictions=gradientBoostingClassifier.predict(x_test)

test_labels = y_test.tolist()
total_num = 0
right_num = 0
for (predict, real) in zip(predictions, test_labels):
	if predict == real:
		right_num = right_num + 1
	total_num = total_num + 1
print("auc:", round(float(right_num)/total_num, 4))


