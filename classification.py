import pandas as pd

update = pd.read_csv("cust_df.csv")
trans = pd.read_csv("transaction_data.csv")
def Rename_OfferTier(df):
    for col in ['CustomerTier']:
        df[col] = df[col].map({'B':'Bronze', 'S':'Saphire','P':'Platinum','G':'Gold'})
    return df
update = Rename_OfferTier(update)
update = update[["CustomerID","CustomerTier"]]
update.rename(columns={'CustomerTier': 'customer_tier',
                         'CustomerID': 'customer_id'}, inplace=True)
print(update.head(10))
joined = update.merge(trans, on = 'customer_id',how ='inner')
print(joined.head())
joined.to_csv("transaction_df.csv",index=False)