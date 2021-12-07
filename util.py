from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import metrics
import pandas as pd

def remove_invalid_rows(df):
  """Removing summary data and selecting records which are either 'Fully Paid' or 'Charged Off'"""
  # Removing summary rows which has empty member ID
  df = df[df.loan_amnt.notnull()]
  # Removing loans other than "Fully Paid" and "Charged Off" status
  df = df[(df.loan_status == 'Fully Paid') | (df.loan_status == 'Charged Off')]

  return df

def convert_to_date(df):
  # Converting to date type
  df["issue_d"] = pd.to_datetime(df['issue_d'], format='%b-%Y')
  df["earliest_cr_line"] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
  return df


def get_data_since(df, date):
  return df[df["issue_d"].dt.date > date]


def add_issue_date_feature(df):
  df.loc[:,'issue_year']  = df['issue_d'].dt.year
  df.loc[:,'issue_month'] = df['issue_d'].dt.month
  return df


def remove_features(df):
  """Removing features that do not have values like id and features
  that are only available after the loan is administered
  """
  # We need to revisit the importance of these features
  # earliest_cr_line is transformed into number of months since earliest credit line feature
  optional_list = ['earliest_cr_line', 'zip_code', 'emp_title', 'desc', 'title', 'addr_state']
  # These features are identifications without any value
  identity = ['id', 'member_id', 'url']
  # These features have the same value for the whole records
  constant = ['policy_code', 'pymnt_plan', 'out_prncp', 'out_prncp_inv']
  # The following features are available only after the loan is issued
  # So, the features won't be available during in production prediction envirnoment
  # making it irrelevant for traning
  debt_settlement = ['debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
                   'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term']
  hardship_program = ['hardship_flag', 'hardship_type', 'hardship_reason',
                      'hardship_status', 'deferral_term', 'hardship_amount',
                      'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
                      'hardship_length', 'hardship_dpd', 'hardship_loan_status',
                      'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
                      'hardship_last_payment_amount']
  fico_change = ['last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low']
  loan_payment = ['last_pymnt_d', 'next_pymnt_d', 'last_pymnt_amnt']
  collection = ['collection_recovery_fee', 'recoveries']
  fund = ['funded_amnt', 'funded_amnt_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int',
          'total_rec_late_fee', 'total_rec_prncp']
  # There is only 1.38% data available for the following features; secondary applicant
  # We belive these won't add any value as the rest of the rows needs to be filled with fake data
  secondary = ['sec_app_fico_range_low', 'sec_app_fico_range_high', 'sec_app_earliest_cr_line',
               'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
               'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
               'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
               'sec_app_mths_since_last_major_derog', 'dti_joint', 'annual_inc_joint',
               'revol_bal_joint', 'verification_status_joint']
  # Merging together all features that will be removed
  features = identity + optional_list + debt_settlement + hardship_program \
             + fico_change + loan_payment + collection + fund + secondary + constant
  df = df.drop(features, axis=1)
  return df


def months_since_earliest_cr_line(df):
  """Converts earliest credit line to months with respective to loan isssue date"""
  # Calculating the months upto issue date
  df["earliest_cr_line_months"] =((df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12 +
                                        (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month))

  return df


def add_unemployment_rate(unemployment, df):
  df['unemployment_rate'] = df.apply(lambda row: unemployment.loc[row["addr_state"],row["issue_d"].strftime('%b-%y')], axis=1)
  return df


# Label encoding
def encode_ordinal(df, features):
  """Encoding ordinal data"""
  scale_mapper = {
      'sub_grade': {},
      'grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F':6, 'G':7},
      'emp_length': {'0 year':1,'< 1 year':2,'1 year':3,'2 years':4,'3 years':5,'4 years':6,
          '5 years':7,'6 years':8,'7 years':9,'8 years':10,'9 years':11,'10+ years':12}
  }
  count = 1
  for grade, val in scale_mapper['grade'].items():
    for i in range(1, 6, 1):
      scale_mapper['sub_grade']["%s%s"%(grade, i)] = count
      count += 1

  for feature, mapper in scale_mapper.items():
    df[feature] = df[feature].replace(mapper)
  
  return df


def split_data(df):
  """
  Split the dataset into 70% training 20% development 10% test data
  The split is done by chronological order
  """
  df = df.sort_values(by='issue_d', ascending=True).reset_index(drop=True)
  train_data, test_data = train_test_split(df, test_size=0.3, shuffle=False)
  dev_data, test_data = train_test_split(test_data, test_size=0.33, shuffle=False)
  dev_data = dev_data.reset_index(drop=True)
  test_data = test_data.reset_index(drop=True)

  train_data = train_data.drop(['issue_d'], axis=1)
  dev_data = dev_data.drop(['issue_d'], axis=1)
  test_data = test_data.drop(['issue_d'], axis=1)
  return (train_data, dev_data, test_data)


def get_ML(df, label):
  """Split the data into Matrix and Label"""
  L = df[label]
  M = df.drop([label], axis=1)
  return M, L


def up_sample(df, label):
  # Separate Charged Off and Fully Paid classes
  data = df.copy()
  data["loan_status"] = label
  data_charged_off = data[label==0]
  data_fully_paid = data[label==1]

  # Up-sampling
  charged_off_upsampled = resample(data_charged_off, replace=True, random_state=0,
                                   n_samples=data_fully_paid.shape[0])

  # Putting back trainig data and label
  new_data = pd.concat([charged_off_upsampled, data_fully_paid])
  new_label = new_data.loan_status
  new_data.drop('loan_status', axis=1, inplace=True)
  return(new_data, new_label)


def get_model_stat(model, train_data, train_label, dev_data, dev_label):
  """Builds metrics for the given model and data"""
  dev_f1_score, dev_accuracy_score, train_f1_score, train_accuracy_score = \
    get_scores(clf, train_data, train_label, dev_data, dev_label)

  return {'Dev F1 Score': dev_f1_score, 'Dev Accuracy Score': dev_accuracy_score,
          'Train F1 Score': train_f1_score, 'Train Accuracy Score': train_accuracy_score}


def get_scores(model, train_data, train_label, dev_data, dev_label):
  """Build different score for a given model and data"""
  dev_predict = model.predict(dev_data)
  train_predict = model.predict(train_data)
  
  dev_f1_score = round(metrics.f1_score(dev_label, dev_predict), 3)
  train_f1_score = round(metrics.f1_score(train_label, train_predict), 3)
  dev_accuracy_score = round(metrics.accuracy_score(dev_label, dev_predict), 3)
  train_accuracy_score = round(metrics.accuracy_score(train_label, train_predict), 3)
  
  return dev_f1_score, dev_accuracy_score, train_f1_score, train_accuracy_score