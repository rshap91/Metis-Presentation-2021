import numpy as np
import pandas as pd
import itertools as it

# python fake_data.py

df = (
    pd.read_csv('static/market_data.csv', parse_dates=['Date'])
      .loc[lambda f: f.Region.ne('all_regions'), ['Date', 'Brand', 'Region', 'Spend']]
      .groupby([pd.Grouper(key='Date', freq='m'), 'Brand', 'Region'])
      .sum()
      .reset_index()
      .assign(Spend= lambda f: f.Spend.map(lambda s: max(s, 0)))
)


# curve
idx = list(it.product(df.Brand.unique(), df.Region.unique()))

# means for each brand
brand_mn = stats.expon(loc=200, scale=500).rvs(df.Brand.nunique())
# b0 for mm curve for each region (but pulled from the distribution for the corresponding brand so they are correlated)
a = stats.norm(loc=brand_mn, scale=50).rvs([4,5]).T.ravel()

# rate parameters are random
b = stats.norm(loc=500, scale=50).rvs(len(idx))
# fuc to generate some noise for each value 
c = stats.halfnorm(loc=50, scale=25).rvs

coefs = zip(a,b)

# create fake target
for i, cat in enumerate(idx):
    mask = (df.Brand==cat[0]) & (df.Region==cat[1])
    # exp
    # df.loc[mask, 'Sales'] = a[i]*df.loc[mask, 'Spend']**b[i] + c(mask.sum())
    # mm
    df.loc[mask, 'Sales'] = (a[i] * df.loc[mask, 'Spend'])/(b[i] + df.loc[mask, 'Spend']) + c(mask.sum())

# Ultimately I figured breaking it out by region would be making it more complicated than an intro presentation should be
#+ and also not enough time so I'm gonna just pretend this is brand only data...
#+ The resulting data will look like it has different "groupings" which an astute practitioner might cluster/relabel and would find they get better models.

# Re-index the dates so they fit in the same space (eg it's monthly data with four rows per brand per date so make it weekly data with one row per brand per date)
ix = np.concatenate([pd.date_range(df.Date.min(), df.Date.max(), 200).floor('d').values for _ in range(5)])
df['Week'] = ix

    
df.drop("Region",1).to_csv('static/market_data_final.csv', index=False)


