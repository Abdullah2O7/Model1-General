#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import janitor


# In[56]:


data=pd.read_excel('Mental disorder symptoms.xlsx')


# In[58]:


mental_data=data.clean_names()
mental_data


# In[60]:


mental_data=mental_data.rename(columns={'ag+1_629e': 'age'})


# In[62]:


mental_data.isna().sum()


# In[64]:


mental_data.duplicated().sum()


# In[66]:


mental_data.head(20)


# In[68]:


mental_data.drop_duplicates(inplace=True)
mental_data.head(20)


# In[70]:


plt.figure(figsize=(5, 5))
plt.scatter(mental_data['age'],mental_data['disorder'])
plt.title('Age Distribution')
plt.show()


# In[71]:


plt.figure(figsize=(15, 10))
sns.countplot(x='disorder', data=mental_data)
plt.title('Disorder Count')
plt.xticks(rotation=45)
plt.show()


# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
mental_data['disorder'] = label_encoder.fit_transform(mental_data['disorder'])

X = mental_data.drop('disorder', axis=1)
X = X.drop('age', axis=1)

y = mental_data['disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[74]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score


# In[78]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)


# In[80]:


svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)


# In[82]:


models = {'RandomForest': rf_predictions, 'SVM': svm_predictions}

for model_name, predictions in models.items():
    print(f'---{model_name}---')
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(f'F1 Score: {f1_score(y_test, predictions, average="weighted")}')
    print(f'Recall: {recall_score(y_test, predictions, average="weighted")}')
    print(classification_report(y_test, predictions))


# In[84]:


performance_data = {
    'Model': ['RandomForest', 'SVM'],
    'Accuracy': [accuracy_score(y_test, rf_predictions), accuracy_score(y_test, svm_predictions)],
    'F1 Score': [f1_score(y_test, rf_predictions, average="weighted"), f1_score(y_test, svm_predictions, average="weighted")],
    'Recall': [recall_score(y_test, rf_predictions, average="weighted"), recall_score(y_test, svm_predictions, average="weighted")]
}

performance_df = pd.DataFrame(performance_data)


# In[86]:


performance_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance')
plt.ylabel('Score')
plt.show()


# In[88]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)


best_params_rf = grid_search_rf.best_params_
best_rf_model = grid_search_rf.best_estimator_


# In[89]:


best_rf_predictions = best_rf_model.predict(X_test)
print(f'---Tuned RandomForest---')
print(f'Accuracy: {accuracy_score(y_test, best_rf_predictions)}')
print(f'F1 Score: {f1_score(y_test, best_rf_predictions, average="weighted")}')
print(f'Recall: {recall_score(y_test, best_rf_predictions, average="weighted")}')
print(classification_report(y_test, best_rf_predictions))


# In[92]:


tuned_performance_data = {
    'Model': ['Tuned RandomForest'],
    'Accuracy': [accuracy_score(y_test, best_rf_predictions)],
    'F1 Score': [f1_score(y_test, best_rf_predictions, average="weighted")],
    'Recall': [recall_score(y_test, best_rf_predictions, average="weighted")]
}

tuned_performance_df = pd.DataFrame(tuned_performance_data)


# In[94]:


tuned_performance_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Tuned Model Performance')
plt.ylabel('Score')
plt.show()


# In[122]:


import numpy as np
import pandas as pd
import pickle

class DisorderPredictor:
    def init(self, pkl_file_path):
        # Load the model pipeline from the pickle file
        with open(pkl_file_path, 'rb') as file:
            self.pipeline = pickle.load(file)
        
        # Initialize encoders and feature list
        self.yes_no_encoder = {'YES': 1, 'NO': 0}
        self.columns = [
            'feeling_nervous', 'panic', 'breathing_rapidly', 'sweating',
            'trouble_in_concentration', 'having_trouble_in_sleeping',
            'having_trouble_with_work', 'hopelessness', 'anger', 'over_react',
            'change_in_eating', 'suicidal_thought', 'feeling_tired', 'close_friend',
            'social_media_addiction', 'weight_gain', 'introvert',
            'popping_up_stressful_memory', 'having_nightmares',
            'avoids_people_or_activities', 'feeling_negative',
            'trouble_concentrating', 'blamming_yourself', 'hallucinations',
            'repetitive_behaviour', 'seasonally', 'increased_energy'
        ]
    
    def preprocess_new_data(self, new_data):
        # Replace and encode YES/NO columns
        for column in self.columns:
            if column in new_data.columns:
                new_data[column] = new_data[column].replace({'YES ': 'YES', ' NO ': 'NO', 'NO ': 'NO', 'YES': 'YES', 'NO': 'NO'})
                new_data[column] = new_data[column].map(self.yes_no_encoder)
        
        # Fill missing columns with 0
        for column in self.columns:
            if column not in new_data.columns:
                new_data[column] = 0
        
        return new_data

    def predict_new_data(self, answers):
        # Convert answers to DataFrame
        new_data = pd.DataFrame([answers], columns=self.columns)
        # Preprocess the data
        new_data_processed = self.preprocess_new_data(new_data)
        # Predict using the model
        predictions = self.pipeline.predict(new_data_processed)
        return predictions[0]

    def decode_result(self, prediction):
        # Map prediction to disorder
        disorders = ['ADHD', 'ASD', 'LONELINESS', 'MDD', 'OCD', 'PDD', 'PTSD', 'ANEXITY', 
                     'BiPolar', 'Eating Disorder', 'Psychotic depression', 'sleeping disorder']
        return disorders[prediction]

    def find_disorder(self, answers):
        # Predict and decode result
        prediction = self.predict_new_data(answers)
        return self.decode_result(prediction)


# In[124]:


with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)

with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[ ]:




