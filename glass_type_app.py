import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, plot_roc_curve, plot_precision_recall_curve,ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feature=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
def prediction(model,feature):
  glass_type=model.predict([feature])
  if glass_type[0]==1:
    return "building windows float processed"
  elif glass_type[0]==2:
    return "building windows non float processed"
  elif glass_type[0]==3:
    return "vehicle windows float processed"
  elif glass_type[0]==4:
    return "vehicle windows non float processed"
  elif glass_type[0]==5:
    return 'containers'
  elif glass_type[0]==6:
    return 'tableware'
  else:
    return 'headlamp'

st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Analysis')

if st.sidebar.checkbox('Show raw data'):
  st.subheader('Full Dataset')
  st.dataframe(glass_df)

st.sidebar.subheader('Scatter Plot')
# Choosing x-axis values for the scatter plot.
feature_list=st.sidebar.multiselect('Select the x-axis values:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)
for i in feature_list:
  st.subheader(f'Scatter plot between glass_type and {i}')
  plt.figure(figsize=(10,7))
  sns.scatterplot(x=glass_df[i],y=glass_df['GlassType'])
  st.pyplot()



st.sidebar.subheader('Visualisation Selector')
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_types=st.sidebar.multiselect('Select the Charts/Plots',('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

if 'Histogram' in plot_types:
    st.subheader('Histogram')
    feature=st.sidebar.selectbox('Select the feature for Histogram:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
    
    st.subheader(f'Histogram for {feature}')
    plt.figure(figsize=(10,7))
    plt.hist(glass_df[feature],bins='sturges',edgecolor='c')
    st.pyplot()
if 'Box Plot' in plot_types:
    st.subheader('Box Plot')
    feature_2=st.sidebar.selectbox('Select the feature for boxplot:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  
    st.subheader(f'Boxplot for {feature_2}')
    plt.figure(figsize=(10,7))
    sns.boxplot(x=glass_df[feature_2])
    st.pyplot()
if 'Count Plot' in plot_types:
    st.subheader('Count Plot')

  
    st.subheader(f'Count plot for glass types')
    plt.figure(figsize=(10,7))
    sns.countplot(x=glass_df['GlassType'])
    st.pyplot()
if 'Pie Chart' in plot_types:
    st.subheader('Pie Chart')
 
  
    st.subheader(f'Pie Chart for glass types')
    plt.figure(figsize=(10,7))
    plt.pie(glass_df['GlassType'].value_counts(),labels=glass_df['GlassType'].value_counts().index,startangle=20,autopct='%1.2f%%')
    st.pyplot()
if 'Correlation Heatmap' in plot_types:
    st.subheader('Correlation Heatmap')

 
  
    
    plt.figure(figsize=(10,7))
    sns.heatmap(glass_df.corr(),annot=True)
    st.pyplot()

if 'Pair Plot' in plot_types:
    st.subheader('Pair Plot')
    plt.figure(figsize=(10,7))
    sns.pairplot(glass_df)
    st.pyplot()

ri=st.sidebar.slider('Enter RI',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na=st.sidebar.slider('Enter Na',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg=st.sidebar.slider('Enter Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al=st.sidebar.slider('Enter Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
Si=st.sidebar.slider('Enter Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
k=st.sidebar.slider('Enter K',float(glass_df['K'].min()),float(glass_df['K'].max()))
ca=st.sidebar.slider('Enter Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
ba=st.sidebar.slider('Enter Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
fe=st.sidebar.slider('Enter Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))

st.sidebar.subheader('Choose Classifier')
# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Random Forest Classifier','Logistic Regression'))

if classifier=='Support Vector Machine':
  st.sidebar.subheader('Model HyperParameters')
  c=st.sidebar.number_input('C',1,100,step=1)
  kernel=st.sidebar.radio('Kernel',('linear','rbf','poly'))
  gamma=st.sidebar.number_input('gamma',1,100,step=1)
  if st.sidebar.button('Classify'):
    model=SVC(C = c, kernel = kernel, gamma = gamma)
    model.fit(X_train,y_train)
    score=model.score(X_train,y_train)
    y_test_pred=model.predict(X_test)
    glass_type=prediction(model,[ri, na, mg, al, Si, k, ca, ba, fe])
    st.write(f'Predicted Glass type is {glass_type} ')
    st.write(f'Accuracy of model is {score}')
    st.write(confusion_matrix(y_test,y_test_pred))
    ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred)
    st.pyplot()

if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, [ri, na, mg, al, Si, k, ca, ba, fe])
        y_test_pred=rf_clf.predict(X_test)
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred)
        st.pyplot()
if classifier == 'Logistic Regression':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)

    max_iter = st. sidebar.number_input("Maximum Iteration", 10, 1000, step = 10)

    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        lr_model=LogisticRegression(C = c_value, max_iter = max_iter)
        lr_model.fit(X_train,y_train)
        y_pred = lr_model.predict(X_test)
        accuracy = lr_model.score(X_test, y_test)
        glass_type = prediction(lr_model, [ri, na, mg, al, Si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
        st.pyplot()



