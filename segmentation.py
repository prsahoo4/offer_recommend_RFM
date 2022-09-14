# Import libraries
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import squarify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# Read dataset
transaction_df = pd.read_csv('transaction_data.csv')
transaction_df.drop(['offer_target_age'], 1, inplace=True)

cust_df = pd.read_csv("Sample_Data_Offer_Customer_Transaction\customer_data.csv")
cust_df.drop(['member_title','segment','member_age','cust_tier'], 1,
             inplace=True)

offer_df = pd.read_csv("Sample_Data_Offer_Customer_Transaction\offer_data.csv",encoding= 'unicode_escape')
offer_df.drop(['targetted_marital_status','targetted_tier','targetted_gender','targetted_age'],1,inplace=True)

#joining the data
joinedData_df = cust_df.merge(transaction_df, on = 'customer_id',how ='inner')

def Rename_OfferTier(df):
    for col in ['customer_tier']:
        df[col] = df[col].map({'B':'Bronze', 'S':'Saphire','P':'Platinum','G':'Gold'})
    return df

joinedData_df = Rename_OfferTier(joinedData_df)
offerTransaction = offer_df.merge(joinedData_df, on='offer_id',how='inner')

# Convert InvoiceDate from object to datetime format
offerTransaction['invoice_date'] = pd.to_datetime(offerTransaction['offer_transaction_date'])

# Drop NA values from online
offerTransaction.dropna()

# --Group data by customer_id--
# Create snapshot date
snapshot_date = offerTransaction['invoice_date'].max() + timedelta(days=1)

# Grouping by CustomerID
data_process = offerTransaction.groupby(['customer_id'],as_index=False).agg({
        'invoice_date': lambda x: (snapshot_date-x.max()).days,
        'offer_id': 'count',
        'transaction_amount': 'sum'})

# Rename the columns
data_process.rename(columns={'invoice_date': 'Recency',
                         'offer_id': 'Frequency',
                         'transaction_amount': 'MonetaryValue'}, inplace=True)

# Print top 20 rows and shape of dataframe
print(data_process.head())

#join customer data with rfm data
cust_df = pd.read_csv("Sample_Data_Offer_Customer_Transaction\customer_data.csv")

data_process = data_process.merge(cust_df,on = 'customer_id',how = 'inner')
data_process.drop(['Unnamed: 0','member_title','segment'],1,inplace = True)
print(data_process.columns.to_list())

# creating a dict file
gender = {'M': 1,'F': 2}
tier = {'B':1,'G':2,'P':3,'S':4}
country = {'AF':1,'AG':2,'AL':3,'AU':4,'GL':5,'IN':6,'US':7,0:0}

#dataframe to number

data_process.cust_tier = [tier[item] for item in data_process.cust_tier]
data_process.member_gender = [gender[item] for item in data_process.member_gender]
data_process= data_process.fillna(0)
data_process.member_country = [country[item] for item in data_process.member_country]



#Visualisation
# data-structure to store Sum-Of-Square-Errors
sse = {}
# Looping over multiple values of k from 1 to 30
for k in range(0, 10):
    kmeans = KMeans(n_clusters=k+1).fit(data_process)
    sse[k] = kmeans.inertia_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster for Recency")
plt.ylabel("SSE")
# Save the Plot in current directory
#plt.show()

#Clustering
kmeans = KMeans(n_clusters=6).fit(data_process)
data_process['cluster'] = kmeans.labels_
print(data_process['cluster'])

data_process.to_csv('clustered.csv',index=False)

# Its important to use binary mode
dbfile = open('clusterPickle', 'ab')

# source, destination
pickle.dump(data_process, dbfile)
dbfile.close()

"""

# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Champions'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Loyal'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Potential'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Promising'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
# Create a new variable RFM_Level
rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)
# Print the header with top 5 rows to the console
print(rfm.head())

# Calculate average values for each RFM_Level, and return a size of each segment
rfm_level_agg = rfm.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']}).round(1)
# Print the aggregated dataset
print(rfm_level_agg)

rfm_level_agg.columns = rfm_level_agg.columns.droplevel()
rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
print(rfm_level_agg)
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfm_level_agg['Count'],
              label=['Can\'t Loose Them',
                     'Champions',
                     'Loyal',
                     'Potential',
                     'Promising','Needs Attention',
                     'Require Activation'], alpha=.6 )
plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()"""