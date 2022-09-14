# Import libraries
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import squarify
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer

# Read dataset
transaction_df = pd.read_excel('Offer_Transaction_Log.xlsx')
transaction_df.drop(['OfferTargetAge'], 1, inplace=True)

cust_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction CEODemo - Copy\Customer_Master.xlsx")
cust_df.drop(['CustomerTitle','CustomerMaritalStatus','CustomerAgeSegment','CustomerAge','CustomerTier'], 1,
             inplace=True)

offer_df = pd.read_excel("Sample_Data_Offer_Customer_Transaction CEODemo - Copy\Offer_Master - Copy.xlsx")
offer_df.drop(['Targeted Marital Status','Targeted Tier','Targeted Type','Targeted Gender','Targeted Preference Group',
               'Targeted Nationality','Targeted Age'],1,inplace=True)

#joining the data
joinedData_df = cust_df.merge(transaction_df, on = 'CustomerID',how ='inner')
joinedData_df.to_csv("Join.csv" , index=False)

def Rename_OfferTier(df):
    for col in ['CustomerTier']:
        df[col] = df[col].map({'B':'Bronze', 'S':'Saphire','P':'Platinum','G':'Gold'})
    return df

joinedData_df = Rename_OfferTier(joinedData_df)

offerTransaction = offer_df.merge(joinedData_df, on='OfferId',how='inner')
offerTransaction.to_csv('offerData.csv',index=False)

# Convert InvoiceDate from object to datetime format
offerTransaction['InvoiceDate'] = pd.to_datetime(offerTransaction['OfferTransactionDate'])

# Drop NA values from online
offerTransaction.dropna()

# --Group data by customerID--
# Create snapshot date
snapshot_date = offerTransaction['InvoiceDate'].max() + timedelta(days=1)
print(snapshot_date)
print(offerTransaction.groupby(['CustomerID']).head())

# Grouping by CustomerID
data_process = offerTransaction.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'OfferId': 'count',
        'TransactionAmount': 'sum'})

# Rename the columns
data_process.rename(columns={'InvoiceDate': 'Recency',
                         'OfferId': 'Frequency',
                         'TransactionAmount': 'MonetaryValue'}, inplace=True)

# Print top 5 rows and shape of dataframe
print(data_process.head(10))

# --Calculate R ,M and F groups--
# Create labels for Recency and Frequency
r_labels = range(4, 0, -1); f_labels = range(1, 5)
# Create labels for MonetaryValue
m_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(data_process['Recency'], q=4, labels=r_labels)
# Assign these labels to 4 equal percentile groups
f_groups = pd.qcut(data_process['Frequency'], q=4, labels=f_labels)
# Assign these labels to three equal percentile groups
m_groups = pd.qcut(data_process['MonetaryValue'], q=4, labels=m_labels)
# Create new columns R , F and M
data_process = data_process.assign(R = r_groups.values, F = f_groups.values,M = m_groups.values)
#print(data_process.head())

# data-structure to store Sum-Of-Square-Errors
sse = {}
# Looping over multiple values of k from 1 to 30
for k in range(0, 10):
    kmeans = KMeans(n_clusters=k+1, random_state=1231).fit(data_process.iloc[:,3:])
    sse[k] = kmeans.inertia_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.show()

model = KMeans(n_clusters=6,random_state=1231).fit(data_process.iloc[:,3:])
centers = model.cluster_centers_
data_process['cluster'] = model.labels_
data_process.to_csv(('datacluster.csv'))
