import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train.csv')

print("Initial Data:")
print(train_df.head())
print("\nData Info:")
print(train_df.info())
print("\nMissing Values:")
print(train_df.isnull().sum())

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df.drop(['Cabin'], axis=1, inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

print("\nStatistical Summary:")
print(train_df.describe(include='all'))

plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='FamilySize', y='Survived', data=train_df)
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=train_df)
plt.title('Age Distribution by Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked_Q', y='Survived', data=train_df)
plt.title('Survival Rate by Embarked Point (Q)')
plt.xlabel('Embarked Point (Q)')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked_S', y='Survived', data=train_df)
plt.title('Survival Rate by Embarked Point (S)')
plt.xlabel('Embarked Point (S)')
plt.ylabel('Survival Rate')
plt.show()
