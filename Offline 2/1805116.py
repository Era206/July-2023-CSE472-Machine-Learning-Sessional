import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample


def telco_preprocessing():
    #data loading from csv file inside dataset folder
    df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    #duplicate data removing
    df.drop_duplicates(inplace=True)
    df.drop(['customerID'], axis=1, inplace=True)
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')


    #train and text data split
    train, test = train_test_split(df, test_size=0.2, random_state=200)



    num_columns = train.select_dtypes(include=['int64', 'float64']).columns
    cat_columns = train.select_dtypes(include=['object']).columns



    col_means=train[num_columns].mean()
    col_stds=train[num_columns].std()
    col_modes=train[cat_columns].mode().iloc[0]

    train[num_columns] = train[num_columns].fillna(col_means)
    train[cat_columns] = train[cat_columns].fillna(col_modes)
    test[num_columns] = test[num_columns].fillna(col_means)
    test[cat_columns] = test[cat_columns].fillna(col_modes)


    for col in cat_columns:
        unique_values_train = train[col].nunique()

        if unique_values_train > 2:  # Use one-hot encoding for columns with more than 2 unique values
            train = pd.get_dummies(train, columns=[col])

        else:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])



    for col in cat_columns:
        unique_values_test = test[col].nunique()

        if unique_values_test > 2:  # Use one-hot encoding for columns with more than 2 unique values
            test = pd.get_dummies(test, columns=[col])

        else:
            le = LabelEncoder()
            test[col] = le.fit_transform(test[col])



    train_col = train.columns
    test_col = test.columns
    missing_cols = set(train.columns) - set(test.columns)

    # Add missing columns to test data with values filled with 0
    for col in missing_cols:
        test[col] = 0

    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train_col]

    # #save train and test data to csv file
    # train.to_csv('train_telco.csv', index=False)
    # test.to_csv('test_telco.csv', index=False)

    # train_x = train.drop(['Churn'], axis=1)
    # train_y = train['Churn']
    # test_x = test.drop(['Churn'], axis=1)
    # test_y = test['Churn']

    # # Create and train the logistic regression model
    # model = LogisticRegression(max_iter=1000)
    # model.fit(train_x, train_y)

    # # Make predictions on the test set
    # pred = model.predict(test_x)

    # # Calculate and print the accuracy of the model
    # accuracy = accuracy_score(test_y, pred)
    # print("Accuracy:", accuracy)
    
    return train, test, 'Churn'


def adult_preprocessing():
    col_names=['age','workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    #load train and test data adult.data and adult.test files respectively
    train = pd.read_csv('dataset/adult.data', names=col_names)
    test = pd.read_csv('dataset/adult.test', names=col_names)
    
    #duplicate data removing
    train.drop_duplicates(inplace=True)
    test.drop_duplicates(inplace=True)

    #check for data types
    test.age = pd.to_numeric(test.age, errors='coerce')

    num_columns = train.select_dtypes(include=['int64', 'float64']).columns
    cat_columns = train.select_dtypes(include=['object']).columns

    col_means=train[num_columns].mean()
    col_stds=train[num_columns].std()
    col_modes=train[cat_columns].mode().iloc[0]

    train[num_columns] = train[num_columns].fillna(col_means)
    train[cat_columns] = train[cat_columns].fillna(col_modes)

    num_columns_test = test.select_dtypes(include=['int64', 'float64']).columns
    cat_columns_test = test.select_dtypes(include=['object']).columns
    
    test[num_columns_test] = test[num_columns_test].fillna(col_means)
    test[cat_columns_test] = test[cat_columns_test].fillna(test[cat_columns_test].mode().iloc[0])

    for col in cat_columns:
        unique_values_train = train[col].nunique()


        if unique_values_train > 2:
            train = pd.get_dummies(train, columns=[col])

        elif unique_values_train == 2:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])

    for col in cat_columns_test:
        unique_values_test = test[col].nunique()

        if unique_values_test > 2:
            test = pd.get_dummies(test, columns=[col])

        else:
            le = LabelEncoder()
            test[col] = le.fit_transform(test[col])

    train_col = train.columns
    test_col = test.columns
    missing_cols = set(train.columns) - set(test.columns)

    # Add missing columns to test data with values filled with 0
    for col in missing_cols:
        test[col] = 0

    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train_col]

    #save train and test data to csv file
    # train.to_csv('train_adult.csv', index=False)
    # test.to_csv('test_adult.csv', index=False)

    # train_x = train.drop(['income'], axis=1)
    # train_y = train['income']
    # test_x = test.drop(['income'], axis=1)
    # test_y = test['income']
    # model=LogisticRegression(max_iter=5000)
    # model.fit(train_x,train_y)
    # pred=model.predict(test_x)
    # accuracy=accuracy_score(test_y,pred)

    # #print recall, specificity, precision, fdr, f1
    # tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()

    # specificity = tn / (tn + fp)
    # precision = precision_score(test_y, pred)
    # recall = recall_score(test_y, pred)  
    # f1 = f1_score(test_y, pred)
    # fdr=1-precision

    # print(".................in built calculation.....................")
    # print("Accuracy:", accuracy)
    # print("recall", recall)
    # print("specificity", specificity)
    # print("precision", precision)
    # print("fdr", fdr)
    # print("f1", f1)

    return train, test, 'income'

def creditcard_preprocessing():
    #data loading from csv file inside dataset folder
    df = pd.read_csv('dataset/creditcard.csv')
    
    df.drop_duplicates(inplace=True)


    train, test = train_test_split(df, test_size=0.2, random_state=200)
    value_counts = train['Class'].value_counts()
    majority_class = value_counts.idxmax()
    
    train_majority = train[train['Class'] == majority_class]
    train_minority = train[train['Class'] != majority_class]
    
    # size_minority = train_minority.shape[0]
    train_majority_undersampled = resample(train_majority, replace=False, n_samples=20000,
                                        random_state=6)
    
    train_balanced = pd.concat([train_majority_undersampled, train_minority])
    train_balanced = train_balanced.sample(frac=1, random_state=6).reset_index(drop=True)
    train=train_balanced

    num_columns = train.select_dtypes(include=['int64', 'float64']).columns
    cat_columns = train.select_dtypes(include=['object']).columns

    col_means=train[num_columns].mean()
    col_stds=train[num_columns].std()
    # col_modes=train[cat_columns].mode().iloc[0]

    train[num_columns] = train[num_columns].fillna(col_means)
    # train[cat_columns] = train[cat_columns].fillna(col_modes)
    test[num_columns] = test[num_columns].fillna(col_means)
    # test[cat_columns] = test[cat_columns].fillna(col_modes)

    for col in cat_columns:
        unique_values_train = train[col].nunique()

        if unique_values_train > 2:  # Use one-hot encoding for columns with more than 2 unique values
            train = pd.get_dummies(train, columns=[col])

        else:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])



    for col in cat_columns:
        unique_values_test = test[col].nunique()

        if unique_values_test > 2:  # Use one-hot encoding for columns with more than 2 unique values
            test = pd.get_dummies(test, columns=[col])

        else:
            le = LabelEncoder()
            test[col] = le.fit_transform(test[col])



    train_col = train.columns
    test_col = test.columns
    missing_cols = set(train.columns) - set(test.columns)

    # Add missing columns to test data with values filled with 0
    for col in missing_cols:
        test[col] = 0

    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train_col]

    #save train and test data to csv file
    # train.to_csv('train_telco.csv', index=False)
    # test.to_csv('test_telco.csv', index=False)

    # train_x = train.drop(['Class'], axis=1)
    # train_y = train['Class']
    # test_x = test.drop(['Class'], axis=1)
    # test_y = test['Class']

    # # Create and train the logistic regression model
    # model = LogisticRegression(max_iter=1000)
    # model.fit(train_x, train_y)

    # # Make predictions on the test set
    # pred = model.predict(test_x)

    # # Calculate and print the accuracy of the model
    # accuracy = accuracy_score(test_y, pred)
    # # print("Accuracy:", accuracy)

    # #print recall, specificity, precision, fdr, f1
    # tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()

    # specificity = tn / (tn + fp)
    # precision = precision_score(test_y, pred)
    # recall = recall_score(test_y, pred)  
    # f1 = f1_score(test_y, pred)
    # fdr=1-precision
  
    # print(".................in built calculation.....................")
    # print("Accuracy:", accuracy)
    # print("recall", recall)
    # print("specificity", specificity)
    # print("precision", precision)
    # print("fdr", fdr)
    # print("f1", f1)

    # print(type(train))
    # print(type(test))

    return train, test, 'Class'


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z.astype(np.float64)))
    
def sigmoid(z):
    z = z.astype(np.float64)  # Ensure using float64 for higher precision if not already
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# def loss_function(y, y_hat):
#     return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def loss_function(y, y_hat):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # Clip y_hat to avoid log(0) or log(1)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))



def train_model(train, output_col, epochs, lr, threshold, best_features_count):
    X=train.drop([output_col], axis=1)
    # X=X.astype(np.float64)
    y=train[output_col]
    y=y.astype(np.float64)

    # write the code to select best features from X using in built Mutual_info_classif function
    if best_features_count > X.shape[1]:
        best_features_count = X.shape[1]
    
    # selector = SelectKBest(mutual_info_classif, k=best_features_count)
    # selected_indices = selector.fit(X, y).get_support(indices=True)
    selector=mutual_info_classif(X,y)

    sorted_selector=sorted(selector,reverse=True)
    selected_columns=[]
    th=sorted_selector[best_features_count-1]
    for i in range(len(selector)):
        if selector[i]>=th:
            selected_columns.append(i)

    

    # Get the names of the top k features
    top_k_features = X.columns[selected_columns].tolist()
    # top_k_features=X.columns[selected_indices]

    # print("Top k features are:", top_k_features)

    X = X[top_k_features]
    X.insert(0, 'bias', 1)
    X=np.array(X)
    X=X.astype(np.float64)

    m,n=X.shape

    # w=np.zeros(n)
    w = np.random.normal(0,0.01,n)

    # training loop
    for epoch in range(epochs):            
        # calculating hypotheses
        y_predicted1 = np.dot(X, w)
        y_predicted = sigmoid(y_predicted1)
        # here confusion
        # y_predicted = y_predicted2.astype(np.float64)

        # calculating gradients of loss with respect to weights w
        dw = np.dot(X.T, (y-y_predicted))
        dw = 1/m*dw

        loss = loss_function(y, y_predicted)            
        # early termination of gradient descent
        if loss < threshold:
            break

        # gradient descent: updating parameters weights w
        w += lr * dw

    return w, top_k_features

def test_model(test, w, top_k_features, output_col):
    X=test.drop(columns=[output_col])
    y=test[output_col]
    # y=y.astype(np.float64)
    y=y.astype(int)

    X = X[top_k_features]
    X = pd.DataFrame(X)
    X.insert(0, 'bias', 1)
    X=np.array(X)
    # X=X.astype(np.float64)
    X=X.astype(int)
    # m,n=X.shape

    y_predicted_1= np.dot(X, w)
    # print(type(y_predicted))
    y_predicted_2= sigmoid(y_predicted_1)
    y_predicted= np.round(y_predicted_2)

    y=np.array(y)
    acc = accuracy_score(y, y_predicted)

    

    # print("type of y: ", type(y))
    # print("type of y_predicted: ", type(y_predicted))

    return y, y_predicted, acc



def adaboosting_train (train_in, output_col, max_iter,epochs, lr, threshold, best_features_count):
    # X=train.drop([output_col], axis=1)
    train=train_in.copy()
    y=train[output_col]
    #turn y to set
    y=np.array(y)
    y=y.astype(np.float64)

    m=train.shape[0]

    #data weight
    data_weight=np.ones(m)/m
    # [print(i) for i in data_weight]
    hypothesis_weight=[]
    hypothesis_inernal_weights=[]
    hypothesis_top_k_features=[]
    # training loop
    i=0


    while True:
        i=i+1
        # print("---------------------------------------", i, "---------------------------------------")
        if i>max_iter:
            break

        # print(i,data_weight)
        # resampling input examples
        indices=np.random.choice(m,m,replace=True,p=data_weight)

        resampled_data=train.iloc[indices]
        # resampled_data = train.sample(n=m, replace=True, weights=data_weight)

        #save resampled data to csv file
        # resampled_data.to_csv('resampled_data.csv', index=False)

        w, top_k_features = train_model(
            resampled_data,
            output_col,
            epochs=epochs,
            lr=lr,
            threshold=threshold,
            best_features_count=best_features_count
        )


        y_main, y_predicted, acc = test_model(train, w, top_k_features, output_col)

        #check the correctness of y using y_predicted and y
     
        y_predicted=np.array(y_predicted)
        y=np.array(y)
        wrong_index=(y_predicted!=y)
        # print("wrong index", wrong_index)
        #print number of TRUE in wrong_index
        # print("number of wrong index", np.sum(wrong_index))
        # print(np.shape(wrong_index))
        # print(np.sum(data_weight))
        error=np.sum(data_weight[wrong_index])

        # print(i,error)
        if error==0:
            error=1e-15
        if error>0.5:
            i=i-1
            continue
        # if error<0.1:
        #     i=i-1
        #     continue

        # print(i,error)
        #update data weight
        correct_index=(y_predicted==y)
        # print("correct index", correct_index)
        # print("sum correct index", np.sum(correct_index))
        data_weight[correct_index]=data_weight[correct_index]*(error/(1-error))
       
        # print("sum data weight", np.sum(data_weight))
        # print("before normalization data weight", data_weight)
        #normalize data weight
        data_weight=data_weight/np.sum(data_weight)
        # print("after sum data weight", np.sum(data_weight))


        # print(i,data_weight)

        #update hypothesis weight
        # print(error)
        v=np.log((1-error)/error)
        # print(v)
        hypothesis_weight.append(v)
        hypothesis_inernal_weights.append(w)
        hypothesis_top_k_features.append(top_k_features)

        # print(hypothesis_weight)

    # hypothesis_weight=hypothesis_weight/np.sum(hypothesis_weight)



    return hypothesis_weight, hypothesis_inernal_weights, hypothesis_top_k_features

def adaboosting_test(test, hypothesis_weight, hypothesis_internal_weights, hypothesis_top_k_features, output_col):
    y=test[output_col]
    y=y.astype(np.float64)
    sum=np.zeros(test.shape[0])

    y_predicted=np.zeros(test.shape[0])
    # print(hypothesis_weight)
    for i in range(len(hypothesis_weight)):
        w=hypothesis_internal_weights[i]
        top_k_features=hypothesis_top_k_features[i]
        y_main,y_predicted,acc=test_model(test,w,top_k_features,output_col)
        

        sum=sum+hypothesis_weight[i]*(2*y_predicted-1)
    

    y_predicted=(sum>0).astype(np.int64)
    acc=accuracy_score(y,y_predicted)
    return y_predicted, acc






def test():
    np.random.seed(42)
    # --------------------Data Preprocessing--------------------


    # --------------------Telco Churn Dataset--------------------
    train, test, output_col = telco_preprocessing()
    # --------------------Adult Dataset--------------------
    # train, test, output_col = adult_preprocessing()
    # --------------------Credit Card Dataset--------------------
    # train, test, output_col = creditcard_preprocessing()

    # --------------------Logistic Regression--------------------

    epochs = 5000
    lr = 0.5
    threshold = 0
    # best feature count for ds_telco=10, ds_adult=5,ds_credit=8
    best_features_count = 10
    # w, top_k_features, acc = logistic_regression(train, test, epochs, lr, threshold, best_features_count, output_col)
    w, top_k_features = train_model(train, output_col, epochs, lr, threshold, best_features_count)

    print(".................for train....................")
    y_true, y_predicted, acc = test_model(train, w, top_k_features, output_col)
    #counting confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)  
    f1 = f1_score(y_true, y_predicted)
    fdr=1-precision
    print("Accuracy:", acc)
    print("recall", recall)
    print("specificity", specificity)
    print("precision", precision)
    print("fdr", fdr)
    print("f1", f1)

    
    print("....................for test....................")
    y_true, y_predicted, acc = test_model(test, w, top_k_features, output_col)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()

    # sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)  
    f1 = f1_score(y_true, y_predicted)
    fdr=1-precision
    # print("sensitivity", sensitivity)
    print("Accuracy:", acc)
    print("recall", recall)
    print("specificity", specificity)
    print("precision", precision)
    print("fdr", fdr)
    print("f1", f1)



    # # --------------------Adaboosting--------------------
    # hypothesis_weight, hypothesis_inernal_weights, hypothesis_top_k_features = adaboosting_train(train, output_col, max_iter=20,epochs=2, lr=0.0001, threshold=0.5, best_features_count=6)
    # k_iter_list=[5,10,15,20]
    # for k in k_iter_list:
    #     print("....................",k,"....................")
    #     hypothesis_weight_k=hypothesis_weight[:k]
    #     hypothesis_inernal_weights_k=hypothesis_inernal_weights[:k]
    #     hypothesis_top_k_features_k=hypothesis_top_k_features[:k]
    #     y_predicted, acc = adaboosting_test(test, hypothesis_weight_k, hypothesis_inernal_weights_k, hypothesis_top_k_features_k, output_col)
    #     y_pred, acc1 = adaboosting_test(train, hypothesis_weight_k, hypothesis_inernal_weights_k, hypothesis_top_k_features_k, output_col)
    #     print("Accuracy in train:", acc1)
    #     print("Accuracy in test:", acc)

    
test()








