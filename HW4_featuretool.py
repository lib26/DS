
import pandas as pd
import numpy as np
import featuretools as ft

# read dataset
clients = pd.read_csv('data/clients.csv')
loans = pd.read_csv('data/loans.csv')
payments = pd.read_csv('data/payments.csv')

# Create new entityset
es = ft.EntitySet(id = 'clients')

# Create an entity from the client dataframe
es = es.add_dataframe(dataframe_name='clients',
                      dataframe=clients,
                      index='client_id',
                      time_index='joined')

# Create an entity from the loans dataframe
es = es.add_dataframe(dataframe_name='loans',
                      dataframe=loans,
                      index='loan_id',
                      time_index='loan_start')

# Create an entity from the payments dataframe
es = es.add_dataframe(dataframe_name='payments',
                      dataframe=payments,
                      make_index=True,
                      index='payment_id',
                      time_index='payment_date')


# Group loans by client id and calculate total of loans
stats = loans.groupby('client_id')['loan_amount'].agg(['sum'])
stats.columns = ['total_loan_amount']

# Merge with the clients dataframe
stats = clients.merge(stats, left_on='client_id', right_index=True, how='left')
print(stats.head(10))










