class ReplaceNaN:
  """It handles nan using custom values"""
  # Constant value replacement
  # There are almost no 0 value which indicates that the Nan is meant to be 0 given that features are not common
  nan_to_zero = [
      'mths_since_last_delinq',         # The number of months since the borrower's last delinquency.
      'mths_since_last_record',         # The number of months since the last public record.
      'mo_sin_old_il_acct',             # Months since oldest bank installment account opened.
      'mths_since_recent_bc_dlq',       # Months since most recent personal finance delinquency.
      'mths_since_recent_revol_delinq', # Months since most recent revolving delinquency.
      'mths_since_last_major_derog'     # Months since most recent 90-day or worse rating.
  ]

  def fit(self, df):
    pass

  def transform(self, df):
    """Replace Nan values"""
    df[self.nan_to_zero] = df[self.nan_to_zero].fillna(value=0)
    df['emp_length'].fillna(value='0 year', inplace=True) # Those are with 0 annual income
    df['dti'].fillna(value=0, inplace=True)               # Those are with 0 annual income
    return df

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)