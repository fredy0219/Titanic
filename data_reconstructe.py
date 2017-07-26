import numpy as np
import pandas as pd

def substrings_in_string(big_string, substrings):
	for substring in substrings:
		if substring in big_string:
			return substring

def getData():
	title_table = ['Mrs','Mr','Miss','Master','Dr',\
	'Rev','Major','Col','Mlle','Jonkheer','Ms','Sir','Don','Mme','Capt','Lady','the Countess']
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

	cabin_table = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
	cabin_mapping = {"A": 1, "B": 2, "C": 3, "D": 4,'E': 5, "F": 6,'G': 7, "T": 8}

	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	Y = df_train['Survived'].as_matrix()

	fare_mode = df_test[df_test['Pclass']==3]['Fare'].mode()
	df_test['Fare'] = df_test.fillna(fare_mode[0])

	emb_mode = df_train[(df_train['Pclass']==1)&(df_train['Fare']<=85)&(df_train['Fare']>75)]['Embarked'].mode()
	df_train['Embarked'] = df_train['Embarked'].fillna(emb_mode[0])

	combine = [df_train , df_test]

	age_mean = (df_train['Age'].sum() + df_test['Age'].sum()) / ( len(df_train['Age'].index) + len(df_test['Age'].index))


	for df in combine:

		#df_train['Title'] = row.astype(str).map(lambda x: substrings_in_string(x, title_table))
		df['Title'] = df['Name'].astype(str).map(lambda x: substrings_in_string(x, title_table))
		df['Title'] = df['Title'].replace(['Major','Rev','Don','Col','Dr','Jonkheer','Sir','Capt','Lady','the Countess'],'Rare')
		df['Title'] = df['Title'].replace('Mlle','Miss')
		df['Title'] = df['Title'].replace('Ms','Miss')
		df['Title'] = df['Title'].replace('Mme','Mrs')
		df['Title'] = df['Title'].map(title_mapping);
		df['Title'] = df['Title'].fillna(0)

		#print df['Title'].shape

		df['Deck'] = df['Cabin'].astype(str).map(lambda x: substrings_in_string(x, cabin_table))
		df['Deck'] = df['Deck'].map(cabin_mapping)
		df['Deck'] = df['Deck'].fillna(0)

		#print df['Deck'].value_counts()

		df['Sex'] = df['Sex'].replace('male',0)
		df['Sex'] = df['Sex'].replace('female',1)

		#print df['Sex'].value_counts()

		# age_median = df['Age'].median(skipna=True)
		# # age_mean = df['Age'].mean(skipna=True)
		# age_std = df['Age'].std(skipna=True)

		# df['Age'] = df['Age'].fillna(age_mean)		
		# df['Age'] = (df['Age'] - df['Age'].mean()) / age_std

		age_avg = df['Age'].mean()
		age_std = df['Age'].std()
		age_null_count = df['Age'].isnull().sum()
		age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
		df['Age'][np.isnan(df['Age'])] = age_null_random_list

		df.loc[ df['Age'] <= 16, 'Age'] 					= 0
		df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
		df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
		df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
		df.loc[ df['Age'] > 64, 'Age']                      = 4

		#print df['Age']

		df['Family'] = df['SibSp'] + df['Parch']
		df['isAlone'] = 0
		df.loc[df['Family']==0, 'isAlone'] = 1

		#df['Family'] = (df['Family'] - df['Family'].mean()) / df['Family'].std()

		df['Embarked'] = df['Embarked'].replace('C',1)
		df['Embarked'] = df['Embarked'].replace('Q',2)
		df['Embarked'] = df['Embarked'].replace('S',3)

		#print df['Embarked']
		# df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()
		# df['Fare'] = df['Fare'].astype(np.float64)

		#Mapping Fare
		df.loc[ df['Fare'] <= 7.91, 'Fare']                          = 0
		df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
		df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
		df.loc[ df['Fare'] > 31, 'Fare']                             = 3  
		df['Fare'] = df['Fare'].astype(np.float64)


		if 'Survived' in df.columns:
			del df['Survived']

		del df['PassengerId'],df['Name'],df['SibSp'],df['Parch'],df['Ticket'],df['Cabin'],df['Embarked']

	train = combine[0]
	test = combine[1]

	X = train.as_matrix()
	Xt = test.as_matrix()

	return X,Y,Xt

	# N,D = X.shape
	# X2 = np.zeros((N,D+2+2+5+8))

	# X2[:,3:6] = X[:,1:4]
	# X2[:,-2:] = X[:,-2:]

	# for n in xrange(N):
	# 	t = int(X[n,0])
	# 	X2[n ,t-1] = 1

	# 	t = int(X[n,4])
	# 	X2[n,6+t] = 1

	# 	t = int(X[n,5])
	# 	X2[n,9+t] = 1

	# 	t = int(X[n,6])
	# 	X2[n,15+t] = 1


	# N,D = Xt.shape
	# X2t = np.zeros((N,D+2+2+5+8))

	# X2t[:,3:6] = Xt[:,1:4]
	# X2t[:,-2:] = Xt[:,-2:]

	# for n in xrange(N):
	# 	t = int(Xt[n,0])
	# 	X2t[n ,t-1] = 1

	# 	t = int(Xt[n,4])
	# 	X2t[n,6+t] = 1

	# 	t = int(Xt[n,5])
	# 	X2t[n,9+t] = 1

	# 	t = int(Xt[n,6])
	# 	X2t[n,15+t] = 1
	# return X2 , Y , X2t






















