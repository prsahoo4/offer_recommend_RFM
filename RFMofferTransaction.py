# Import libraries
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
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
data_process.to_csv('data.csv')
print('{:,} rows; {:,} columns'
      .format(data_process.shape[0], data_process.shape[1]))

#Plot RFM distributions
plt.figure(figsize=(12,10))
# Plot distribution of R
plt.subplot(3, 1, 1); sns.histplot(data_process['Recency'])
# Plot distribution of F
plt.subplot(3, 1, 2); sns.histplot(data_process['Frequency'])
# Plot distribution of M
plt.subplot(3, 1, 3); sns.histplot(data_process['MonetaryValue'])
# Show the plot
plt.show()

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
print(data_process.head())

# Concat RFM quartile values to create RFM Segments
def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])
data_process['RFM_Segment_Concat'] = data_process.apply(join_rfm, axis=1)
rfm = data_process
print(rfm.head())

# Count num of unique segments
rfm_count_unique = rfm.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
print(rfm_count_unique.sum())

# Calculate RFM_Score
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)
print(rfm['RFM_Score'].head())

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
plt.show()