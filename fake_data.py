import numpy as np
import pandas as pd
import itertools as it

df = (
    pd.read_csv('static/market_data.csv', parse_dates=['Date'])
      .loc[lambda f: f.Region.ne('all_regions'), ['Date', 'Brand', 'Region', 'Spend']]
      .groupby([pd.Grouper(key='Date', freq='m'), 'Brand', 'Region'])
      .sum()
      .reset_index()
      .assign(Spend= lambda f: f.Spend.map(lambda s: max(s, 0)))
)

ix = np.concatenate([pd.date_range(df.Date.min(), df.Date.max(), 200).floor('d').values for _ in range(5)])
df['Week'] = ix

# curve
idx = list(it.product(df.Brand.unique(), df.Region.unique()))

# Hierarchical for `a` param
# for brand
brand_mn = stats.expon(loc=200, scale=500).rvs(df.Brand.nunique())
# for each region
a = stats.norm(loc=brand_mn, scale=50).rvs([4,5]).T.ravel()

# rest
b = stats.norm(loc=500, scale=50).rvs(len(idx))
# for each val
c = stats.halfnorm(loc=50, scale=25).rvs

coefs = zip(a,b)

# create fake target
for i, cat in enumerate(idx):
    mask = (df.Brand==cat[0]) & (df.Region==cat[1])
    # exp
    # df.loc[mask, 'Sales'] = a[i]*df.loc[mask, 'Spend']**b[i] + c(mask.sum())
    # mm
    df.loc[mask, 'Sales'] = (a[i] * df.loc[mask, 'Spend'])/(b[i] + df.loc[mask, 'Spend']) + c(mask.sum())
    
df.to_csv('static/market_data_final.csv', index=False)