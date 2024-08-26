import streamlit as st
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

st.write("## Hyperparameter Tuning with Scikit-Learn! ")
st.write("Tune **parameter** with Scikit-Learn's *GridSearchCV*")
st.write("\n")

col1, col2 = st.columns(2)
about_expander = col1.expander('About')
about_expander.info('''
                    This web application is a simple demonstration of Hyperparameter tuning with 
                    **GridSearchCV**. The parameters customizable in this app are only limited 
                    and the algorithms and datasets used are from Scikit learn. There may be other combinations 
                    of parameters and algorithms that can yield a better accuracy score for a given dataset.
                     ''')

info_expander = col2.expander('What is Hyperparameter Tuning?')
info_expander.info('''
                    **Hyperparameters** are the parameters that describe the model architecture and 
                    **hyperparameter tuning** is the method of looking for the optimal model architecture
                   ''')

st.sidebar.header("Select Dataset")
dataset_name = st.sidebar.selectbox('Select Dataset', ['Iris Flower', 'Wine Recognition'])
st.write("\n")
st.write(f"### **{dataset_name} Dataset**")

model_name = st.sidebar.selectbox('Pick a model', ['Random Forest', 'KNN', 'Logistic Regression', 'SVM'])

cv_count = st.sidebar.slider('Cross-validation count', 2, 5, 3)
st.sidebar.write('---')
st.sidebar.header("User Input Parameters")

def get_dataset(name):
    df = None
    if name == "Iris Flower":
        df = datasets.load_iris()
    elif name == "Wine Recognition":
        df = datasets.load_wine()
    X = df.data
    Y = df.target
    return X, Y

X, Y = get_dataset(dataset_name)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

st.write('Shape of dataset: ', X.shape)
st.write('Number of classes: ', len(np.unique(Y)))

def get_classifier(model_name):
    clf = None
    parameters = None

    if model_name == 'Random Forest':
        st.sidebar.subheader('Number of Estimators')
        st.sidebar.write("The number of trees in the forest.")
        n1 = st.sidebar.slider('n_estimator1', 1, 40, 5)
        n2 = st.sidebar.slider('n_estimator2', 41, 80, 50)
        n3 = st.sidebar.slider('n_estimator1', 81, 120, 100)
        st.sidebar.write("\n")

        st.sidebar.subheader('Max depth')
        st.sidebar.write("The maximum depth of the tree. If None, then nodes are expanded until all leaves aer pure.")
        md1 = st.sidebar.slider('max_depth1', 1, 7, 1)
        md2 = st.sidebar.slider('max_depth2', 8, 14, 10)
        md3 = st.sidebar.slider('max_depth3', 15, 20, 20)
        
        parameters = {'n_estimators':[n1, n2, n3], 'max_depth':[md1, md2, md3]}
        clf = RandomForestClassifier()
        
    elif model_name == 'SVM':
        st.sidebar.subheader('Kernel Type')
        st.sidebar.write("Specifies the kernel type to be used in the algorithm.")
        kernel_type = st.sidebar.multiselect(
            "",
            options=['liner', 'rbf', 'poly', 'sigmoid'],
            default=['liner', 'rbf', 'poly']
        )
        st.sidebar.write("\n")

        st.sidebar.subheader('Regularization Parameter')
        st.sidebar.write("The strength of the regularization is inversely proportional to C.")
        c1 = st.sidebar.slider('C1', 1, 7, 1)
        c2 = st.sidebar.slider('C2', 8, 14, 10)
        c3 = st.sidebar.slider('C5', 15, 20, 20)
        st.sidebar.write("\n")

        st.sidebar.subheader('Gamma')
        st.sidebar.write("Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.")
        g1 = st.sidebar.slider('gamma1', 0.001, 0.01, 0.001)
        g2 = st.sidebar.slider('gamma2', 0.01, 0.1, 0.01)
        g3 = st.sidebar.slider('gamma3', 0.1, 1.0, 1.0)

        parameters = {'kernel' : kernel_type,
                      'C' : [c1, c2, c3],
                      'gamma': [g1, g2, g3]}
        clf = SVC()

    elif model_name == 'Logistic Regression':
        st.sidebar.subheader('Penalty')
        st.sidebar.write("Used to specify the norm used in the penalization.")
        penalty_type = st.multiselect(
            "",
            options= ['l1', 'l2', 'elasticnet'],
            default= ['l1', 'l2']
        )
        st.sidebar.write("\n")

        st.sidebar.subheader('Regularization Parameter')
        st.sidebar.write("Inverse of regularization strength; must be a positive float.")
        c1 = st.sidebar.slider('C1', 0.01, 1.0, 0.1)
        c2 = st.sidebar.slider('C2', 2, 19, 10)
        c3 = st.sidebar.slider('C3', 20, 100, 80, 10)

        parameters = {'penalty' : penalty_type,
                      'C': [c1, c2, c3]}
        clf = LogisticRegression()
    
    else:
        st.sidebar.subheader('Number of Neighbors (K)')
        st.sidebar.write("Number of neighbors to use by default for `kneighbors` queries.")
        k1 = st.sidebar.slider('n_neighbors1', 1, 5, 3)
        k2 = st.sidebar.slider('n_neighbors2', 6, 10, 5)
        k3 = st.sidebar.slider('n_neighbors3', 11, 15, 13)
        parameters = {'n_neighbors' : [k1, k2, k3]}
        clf = KNeighborsClassifier()

    return clf, parameters

clf, parameters = get_classifier(model_name)
grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=cv_count, return_train_score=False)
grid_search.fit(X, Y)

df = pd.DataFrame(grid_search.cv_results_)

st.header('Tuning Results')
result_df = st.multiselect('', options= ['mean_fit_time', 'std_fit_time', 'mean_score_time',
                                         'std_score_time', 'split0_test_score', 'split1_test_score',
                                         'split2_test_score', 'std_test_score', 'rank_test_score', 'params', 'mean_test_score'],
                            default= ['mean_score_time', 'std_score_time', 'split0_test_score', 'split1_test_score',
                                         'split2_test_score'])

df_result = df[result_df]
st.write(df_result)

st.subheader('Parameters and Mean test score')
st.write(df[['params', 'mean_test_score']])
st.write("Best Score: ", grid_search.best_score_)
st.write("Best Parameters: ", grid_search.best_params_)
