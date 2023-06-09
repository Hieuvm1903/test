

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import folium

from streamlit_folium import st_folium, folium_static
import warnings
warnings.filterwarnings('ignore')
st.title('Decision Support System: Road Safety in London')

DATE_COLUMN = 'date'

DATA_URL = (r"2021_data_accident.csv")



@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
data_load_state = st.text('Loading data...')
data = load_data()
data = data.dropna(how='any',axis=0) 
data_load_state.text("Done!")
dataset = data
City_of_London = dataset[dataset['local_authority_highway'] == 'E09000001']
Westminster = dataset[dataset['local_authority_highway'] == 'E09000033']

data = pd.concat([City_of_London, Westminster], axis = 0)
vis = data.loc[:,['date','time','longitude','latitude']]
vis['time']= pd.to_datetime(vis['time'],format= '%H:%M' )

data2 = vis




from geopy.distance import great_circle

def greatcircle(x,y):
    lat1, long1 = x[0], x[1]
    lat2, long2 = y[0], y[1]
    dist = great_circle((lat1,long1),(lat2,long2)).meters
    return dist

from sklearn.cluster import DBSCAN as dbscan



with st.sidebar:
    
    date1 = st.date_input("From",key = "start",min_value  = datetime.date(2012, 1, 1), max_value = datetime.date.today(), value =  datetime.date(2012, 1, 1) )
    date2 = st.date_input("To",key = "finish",min_value  = date1, max_value = datetime.date.today())
    if st.checkbox('Add time element'):
        hour_to_filter = st.slider('From this hour', 0, 23, 0, key = 2)
        hour_end = st.slider('To this hour', 0, 24, 24,key = 1)
        st.subheader('Map of all accidents from %s:00 to %s:00' % (hour_to_filter, hour_end))

    else:
        hour_to_filter, hour_end = [0,24]
# Some number in the range 0-23

filtered_data = data2[(data2['time'].dt.hour >= hour_to_filter) & (data2['time'].dt.hour < hour_end)
&(data2['date'].dt.date >=date1) & (data2['date'].dt.date <= date2)]

import pyodbc 
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance

conn =  pyodbc.connect(
    Trusted_Connection='Yes',
    Driver='{ODBC Driver 17 for SQL Server}',
    Server='EVOL',
    Database='BKLIGHT'
)
cursor = conn.cursor()
cursor.execute("SELECT * FROM account")
rows = cursor.fetchall()


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    for row in rows:
        st.write(row)
    

# Using object notation
st.write(len(filtered_data.index))

dataset = filtered_data
max_cluster_distance = st.slider('Max distance', 10, 200, 150,key = 'ep',step = 5)
min_samples_in_cluster = st.slider('Min number', 2, 20, 12,key = 'minx')

location_data = dataset[['latitude','longitude']]

clusters = dbscan(eps = max_cluster_distance, min_samples = min_samples_in_cluster, metric=greatcircle).fit(location_data)

labels = clusters.labels_
unique_labels = np.unique(clusters.labels_)

dataset['Cluster'] = labels
# Using "with" notation
location = dataset['latitude'].mean(), dataset['longitude'].mean()

map_plot = folium.Map(location=location,zoom_start=13)
clust_colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
folium.TileLayer('cartodbpositron').add_to(map_plot)
for i in range(0,len(dataset)):
    colouridx = dataset['Cluster'].iloc[i]
    if colouridx == -1:
        pass
    else:
        col = clust_colours[colouridx%len(clust_colours)]
        folium.CircleMarker([dataset['latitude'].iloc[i],dataset['longitude'].iloc[i]], radius = 3, color = col, fill = col).add_to(map_plot)
        
st.divider()  # 👈 Draws a horizontal rule
st.subheader('Cluster accidents hotspot by location')
folium_static(map_plot)


import matplotlib.pyplot as plt
import seaborn as sns


road_cond = data['road_surface_conditions'].value_counts()

road_cond_arr = data['road_surface_conditions'].unique()
road_num_acc_arr = road_cond.values

weather_cond = data['weather_conditions'].value_counts() 

weather_cond_arr = data['weather_conditions'].unique()
weather_num_acc_arr = weather_cond.values
st.subheader('Accident Rate by each Factors')

plt.figure(figsize = (20,9),facecolor='grey')
plt.subplot(1, 2, 1)

plt.pie(road_num_acc_arr, labels = road_cond_arr, colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},explode=(0.02,0.02,0.02,0.02,0.3),autopct="%1.1f%%")
plt.legend()
plt.title("Accident Rate by Road Conditions",weight="bold")


plt.subplot(1, 2, 2)

plt.pie(weather_num_acc_arr, labels = weather_cond_arr,colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},explode=(0.01,0.01,0.01,0.01,0.01,0.20,0.3,0.50,0.7),autopct="%1.1f%%")
plt.legend(loc ="lower left")
plt.title("Accident Rate by Weather Conditions",weight="bold")

fig = plt.gcf()

st.pyplot(fig)

light_cond = data['light_conditions'].value_counts()

light_cond_arr = data['light_conditions'].unique()
light_num_acc_arr = light_cond.values

road_type = data['road_type'].value_counts() 

road_type_arr = data['road_type'].unique()
road_type_num_acc_arr = road_type.values

plt.figure(figsize = (20,9),facecolor='grey')
plt.subplot(1, 2, 1)

plt.pie(light_num_acc_arr, labels = light_cond_arr, colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},autopct="%1.1f%%")
plt.legend()
st.divider()  # 👈 Draws a horizontal rule

plt.title("Accident Rate by Light Conditions",weight="bold")


plt.subplot(1, 2, 2)

plt.pie(road_type_num_acc_arr, labels =road_type_arr,colors = sns.color_palette(),startangle = 30,textprops={'size': 'large'},autopct="%1.1f%%")
plt.legend(loc ="lower left")
plt.title("Accident Rate by Road Type",weight="bold")

fig = plt.gcf()

st.pyplot(fig)
st.divider() 
df_uk = data

fig = plt.figure(figsize=(10,10))
sns.countplot(df_uk,x="speed_limit")
st.pyplot(fig)
st.divider() 
st.subheader('Accidents by time')
fig = plt.figure(figsize=(10, 8))
sns.countplot(data,x="day_of_week")

st.pyplot(fig)
st.divider()  # 👈 Draws a horizontal rule


acc_by_time = dataset.time
acc_by_time['hour']  = pd.to_datetime(acc_by_time.values,format= '%H:%M' ).hour
fig = plt.figure(figsize=(10, 8))

sns.countplot(acc_by_time,x="hour")
st.pyplot(fig)
st.divider() 


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import log_loss

accidents = data
accident_ml = accidents.drop('accident_severity' ,axis=1)
accident_ml = accident_ml[['did_police_officer_attend_scene_of_accident','day_of_week' , 'weather_conditions' , 'road_surface_conditions', 'light_conditions','speed_limit']]

# Split the data into a training and test set.
X_train, X_test, y_train, y_test = train_test_split(accident_ml.values, accidents['accident_severity'].values,test_size=0.20, random_state=99)
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_test, y_test)
acc_random_forest1 = round(random_forest.score(X_test, y_test) * 100, 2)

sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)


from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [4, 5],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300]
}
# Create a based model
random_f = RandomForestClassifier()
# Instantiate the grid search model
grid_search = RandomizedSearchCV(estimator = random_f, param_distributions = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)
fig = plt.figure(figsize=(12,6))
feat_importances = pd.Series(random_forest.feature_importances_, index=accident_ml.columns)
feat_importances.nlargest(5).plot(kind ='bar')
st.subheader('Factor importance')
st.pyplot(fig)
st.divider() 
