# author by lifulong
# first version
# auc = 0.7996
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

train_file="./../data/adult.data"
test_file="./../data/adult.test"

# feature_list
feature_list=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
feature_type = ["continuous", "discreate", "continuous", "discreate", "continuous", "discreate", "discreate", "discreate", "discreate", "discreate", "continuous", "continuous", "continuous", "discreate", "label"]

train_dataset=pd.read_csv(train_file, sep=',', header=None, names=feature_list).values
test_dataset=pd.read_csv(test_file, sep=',', header=None, names=feature_list).values

discreate_features = [ index for (index,typ) in enumerate(feature_type) if typ == 'discreate' ]
continuous_features = [ index for (index,typ) in enumerate(feature_type) if typ == 'continuous' ]
label_features = discreate_features[:]
label_features.append(14)

labelEncoder=LabelEncoder()
labelBinarizer=LabelBinarizer()
oneHotEncoder=OneHotEncoder(handle_unknown='ignore', sparse=True)
#oneHotEncoder=OneHotEncoder(handle_unknown='ignore', sparse=True, categorical_features=discreate_features)

for label in label_features:
	label_feature = np.vstack((train_dataset[:,label].reshape(-1,1), test_dataset[:,label].reshape(-1,1)))
	labelEncoder.fit(label_feature)
	train_dataset[:,label]=labelEncoder.transform(train_dataset[:,label])
	test_dataset[:,label]=labelEncoder.transform(test_dataset[:,label])

continuous_train_data=None
continuous_test_data=None
first_feature=True
for feature in continuous_features:
	if first_feature == True:
		continuous_train_data = train_dataset[:,feature]
		continuous_test_data = test_dataset[:,feature]
		first_feature=False
	else:
		continuous_train_data = np.vstack((continuous_train_data, train_dataset[:,feature]))
		continuous_test_data = np.vstack((continuous_test_data, test_dataset[:,feature]))

continuous_train_data = np.transpose(continuous_train_data)
continuous_test_data = np.transpose(continuous_test_data)
#print(continuous_train_data.shape)
#print(continuous_test_data.shape)

discreate_train_data=None
discreate_test_data=None
first_feature=True
for feature in discreate_features:
	full_feature = np.vstack((train_dataset[:,feature].reshape(-1, 1), test_dataset[:,feature].reshape(-1, 1)))
	#full_feature = train_dataset[:,feature].reshape(-1, 1)
	oneHotEncoder.fit(full_feature)
	if first_feature == True:
		discreate_train_data = oneHotEncoder.transform(train_dataset[:,feature].reshape(-1, 1)).toarray()
		discreate_test_data = oneHotEncoder.transform(test_dataset[:,feature].reshape(-1, 1)).toarray()
		first_feature=False
	else:
		discreate_train_data = np.hstack((discreate_train_data, oneHotEncoder.transform(train_dataset[:,feature].reshape(-1, 1)).toarray()))
		discreate_test_data = np.hstack((discreate_test_data, oneHotEncoder.transform(test_dataset[:,feature].reshape(-1, 1)).toarray()))
	#print("discreate_train_data.shape:", discreate_train_data.shape)
	#print("discreate_test_data.shape:", discreate_test_data.shape)

#print("discreate_train_data.shape:", discreate_train_data.shape)
#print("discreate_test_data.shape:", discreate_test_data.shape)

full_train_dataset = np.hstack((continuous_train_data, discreate_train_data))
full_test_dataset = np.hstack((continuous_test_data, discreate_test_data))
#print("full_train_dataset.shape:", full_train_dataset.shape)
#print("full_test_dataset.shape:", full_test_dataset.shape)

y_train = train_dataset[:,14].astype('int')
y_test = test_dataset[:,14].astype('int')
x_train = full_train_dataset
x_test = full_test_dataset

logisticRegression=LogisticRegression()

logisticRegression.fit(x_train, y_train)

predictions=logisticRegression.predict(x_test)

test_labels = y_test.tolist()
total_num = 0
right_num = 0
for (predict, real) in zip(predictions, test_labels):
	if predict == real:
		right_num = right_num + 1
	total_num = total_num + 1
print("auc:", round(float(right_num)/total_num, 4))


