import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
df = pd.read_csv('heart-disease.csv')
df = pd.read_csv('C:/Users/pc/Desktop/KABBAJ DOC/EDUCATIONNEL/data challenge/Heart_disease/heart-disease.csv')
X = df.drop('target', axis=1)
y= df['target']
# EDA 
print(df.head(10))
print(df.describe())
print(df.info())
# DATA VIZ
# ALL THE DATA
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
df.hist(figsize=(10, 8), bins=30, edgecolor='black')
plt.show()
# data viz of categorical variable
# Create a countplot
# SEX
ax = sns.countplot(x='sex', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',(p.get_x() + p.get_width() / 2., p.get_height()))
plt.title('Count of Sex')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
# CP
ax = sns.countplot(x='cp', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',(p.get_x() + p.get_width() / 2., p.get_height()))
plt.title('Count of Each Category in Column of cp variable')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

#sns.countplot(x='cp', data=df)
#plt.title('Count of Each Category in Column of cp variable')
#plt.show()
# SLOPE / SLOPE vs HD
ax= sns.countplot(x='slope', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',(p.get_x() + p.get_width() / 2., p.get_height()))
plt.title('Count of Each Category in Column of slope variable')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

sns.countplot(x='slope', hue='target', data=df)
plt.show()
# THAL
sns.countplot(x='thal', data=df)
plt.title('Count of Each Category in Column of thal variable')
plt.show()
ax = sns.countplot(x='thal', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',(p.get_x() + p.get_width() / 2., p.get_height()))
plt.title('Distribution of thal variable')
plt.xlabel('category')
plt.ylabel('count')
plt.show()
sns.countplot(x='thal', hue='target', data=df)
plt.show()
# TARGET VARIABLE
ax = sns.countplot(x='target', data=df)  
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',(p.get_x() + p.get_width() / 2., p.get_height()))
plt.title('Distribution of Target Variable')
plt.ylabel('Count')
plt.show()
# Age vs HD 
# age with interval 
bins = [25, 40, 45, 50, 55,60,65,80]
labels = ['25-40', '40-45', '45-50', '50-55','55-60','60-65','65-80']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
sns.countplot(x='age_group', hue='target', data=df)
plt.show()
# trestbps vs HD
bins = [90,100,110,120,125,130,140,180,200]
labels = ['90-100','100-110','110-120','120-125','125-130','130-140','140-180','180-200']
df['trestbps_group']= pd.cut(df['trestbps'], bins= bins)
sns.countplot(x='trestbps_group', hue='target', data=df)
plt.show()
# correlation
#corr_matrix = dff.corr()
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Correlation Matrix')
#plt.show()
# khi2 test
#chi2_stat, p_value, dof, expected = chi2_contingency(pd.crosstab(df['target'], df['sex']))
#print(chi2_stat, p_value, dof, expected)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
# RANDOM FOREST TECHNIQUES 
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Accuracy for random forest:", accuracy)
print("Classification Report:\n", classification_report_str)
# LOGISTIC REGRESSION
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Accuracy logistic regression:", accuracy_lr)
print("Classification Report:\n", classification_report_str)
# coefiscient of logistic regression and their p-value
coefficients = lr.coef_
print("Coefficients estimés :", coefficients)
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
print(result.summary())
odds_ratios = np.exp(result.params)
print("Odds Ratios :")
print(odds_ratios)
# Calculer les intervalles de confiance des odds ratios
conf_int = result.conf_int()
conf_int.columns = ['2.5%', '97.5%']  # Renommer les colonnes
conf_int_odds = np.exp(conf_int)
print("Intervalles de confiance des Odds Ratios :")
print(conf_int_odds)
# ANN 
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann.fit(X_train,y_train)
y_pred = ann.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Accuracy of ANN:", accuracy_ann)
print("Classification Report:\n", classification_report_str)
# DECISION TREES
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
print("Accuracy of decision tree:", accuracy_dt)
print("Classification Report:\n", classification_report_str)
# SVM 
sup_vm = svm.SVC(random_state=42)
sup_vm.fit(X_train,y_train)
y_pred = sup_vm.predict(X_test)
accuracy_svm = accuracy_score(y_test,y_pred)
lassification_report_str = classification_report(y_test, y_pred)
print("accuracy of SVM/SVC: ", accuracy_svm)

