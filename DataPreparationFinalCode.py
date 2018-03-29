
# coding: utf-8

# # Step 1: Data Exploration

# Import the dataset FinalRaw.csv as data

# In[4]:

import pandas as pd
get_ipython().magic(u'matplotlib inline')
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:

data = pd.read_csv("FinalRaw.csv")


# View the data

# In[3]:

print len(data)
data.head()


# Produce one new variable GRP_AFFIN_GRP using AFFIN_GRP by combining all categories other than "AFFRV", "AGNCY", and "BDMKT" to a new category “Others”.

# In[4]:

data.groupby('affin_grp')['affin_grp'].agg({'Frequency':'count'})


# In[5]:

data['grp_affin_grp'] = data['affin_grp'].fillna('Other')
clean_affin_grp = {'ALLY' : 'Other' , 'AFFAUxRLC' : 'Other' , 'GMREL' : 'Other' ,
                   'FF' : 'Other' , 'RLC' : 'Other' , 'INTER' : 'Other'}
data['grp_affin_grp'].replace(clean_affin_grp, inplace = True)
data.groupby('grp_affin_grp')['grp_affin_grp'].agg({'Frequency':'count'})


# Produce a vertical bar char using SGPLOT on this newly created variable.

# In[6]:

n_obs = float(len(data))
grp_affin_grp_pct = {'grp_affin_grp': ['AFFRV','AGNCY','BDMKT','Other'],
                     'percentage':[(len(data[data['grp_affin_grp'] == 'AFFRV']))/n_obs*100,
                                   (len(data[data['grp_affin_grp'] == 'AGNCY']))/n_obs*100,
                                   (len(data[data['grp_affin_grp'] == 'BDMKT']))/n_obs*100,
                                   (len(data[data['grp_affin_grp'] == 'Other']))/n_obs*100]}

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
with sns.color_palette("Paired"):
    plt = sns.barplot(x ='grp_affin_grp', y ='percentage', data = grp_affin_grp_pct)
    plt.set(xlabel = 'grp_affin_grp', ylabel = 'percentage')


# Produce a histogram with kernel density line imposed using SGPLOT on the variable total_fee.

# In[7]:

sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})
with sns.color_palette('Paired'):
    plt = sns.distplot(data['total_fee'], bins=50, kde=True)
    plt.set(xlabel = 'total_fee', ylabel='frequency', title = 'total_fee histogram with kde')


# # Step 2: Categorical Variables Recoding

# Fix affin_grp: Reduce Categories to four categories "AFFRV", "AGNCY", "BDMKT", and “OTHERS”. 

# In[8]:

data['affin_grp']= data['grp_affin_grp'] 
data['affin_grp'].unique()


# Remove GRP_AFFIN_GRP.

# In[9]:

data = data.drop('grp_affin_grp', axis = 1)


# Create Three Missing Value Indicators:
# * MI_PIP_limit = "Missing Indicator: PIP_Limit"
# * MI_CP_limit = "Missing Indicator: CP_Limit"
# * MI_Others = "Missing Indicator: All Other Categorical Variables with Missing"

# In[260]:

data['MI_PIP_limit'] = data['PIP_Limit'].fillna(1)
data.loc[data['MI_PIP_limit'] != 1] = 0

data['MI_CP_limit'] = data['CP_Limit'].fillna(1)
data.loc[data['MI_CP_limit'] != 1] = 0


# In[261]:

catCols = ['BI_Limit', 'CreditActionCode', 'DistributionChannelName', 
           'GoverningStateCode', 'HomeownerInd', 'MarketingPlanCode',
          'PaymentMethodName', 'PaymentPlanDesc', 'PreferredMailDocumentsCode',
          'PriorCarrierTypeCode', 'ProductVersionName', 'StateGroup',
          'StateRegion', 'affin_grp', 'assoc_grp', 'paymentmethodlong',
          'premcat']

data['MI_Others'] = data[catCols].isnull().any(axis=1)
bool_map = {True : 1, False: 0}
data['MI_Others'].replace(bool_map, inplace = True)
data['MI_Others'].unique()


# **Mode Imputation** for Categorical Variables: BIGroupNum , HomeownerInd , PreferredMailDocumentsCode, and PriorCarrierTypeCode

# In[262]:

data['BIGroupNum'].fillna(data['BIGroupNum'].mode(), inplace=True)
data['HomeownerInd'].fillna(data['HomeownerInd'].mode(), inplace=True)
data['PreferredMailDocumentsCode'].fillna(data['PreferredMailDocumentsCode']                                           .mode(), inplace=True)
data['PriorCarrierTypeCode'].fillna(data['PriorCarrierTypeCode'].mode()
                                    , inplace=True) 


# Drop variables: assoc_grp, MarketingPlanCode, GoverningStateCode

# In[263]:

data = data.drop('assoc_grp', axis = 1)
data = data.drop('GoverningStateCode', axis = 1)
data = data.drop('MarketingPlanCode', axis = 1)


# In[10]:

data['PaymentPlanDesc'].unique()


# Split PaymentPlanDesc into two numerical variables N_Payment, DownPayPct 

# In[264]:

#if plan is missing, then dp = 100 and n = 0
#if plan is better budget then dp = 0 and n = 1
#if plan is pay in full then dp = 100 and n = 0
data['PaymentPlanDesc'].fillna('100% Down, 0 Payments')
pay_plan_map = {'Pay in Full' : '100% Down, 0 Payments', 
                'Better Budget': '0% Down, 1 Payments',
                0 : '100% Down, 0 Payments'}
data['PaymentPlanDesc'].replace(pay_plan_map, inplace = True)
data['DownPayPct'] = data['PaymentPlanDesc'].str.split(', ').str[0] .str.split(' ').str[0].str.split('%').str[0]
data['N_Payment'] = data['PaymentPlanDesc'].str.split(', ').str[1] .str.split(' ').str[0]


# For **ProductVersionName**, we can see that we need to combine low frequency categories

# In[265]:

pay_plan_map = {'NY Direct no tier' : 'NY RAD','NY non-RAD Agency': 'NY RAD',
                'RAD5LowCost' : 'RAD5' , 'NCAA' : 'NC non-RAD', 
                '4 Tier' : '2 Tier'}
data['ProductVersionName'].replace(pay_plan_map, inplace = True)


# For **CreditActionCode**, take the first letter of the CreditActionCode

# In[266]:

data['FL_CreditActionCode'] = data['CreditActionCode'].str.replace('R','S'). str.slice(0,1,1)
data['FL_CreditActionCode'].unique()


# Obtain **logit transformation** of the following variables StateGroup, ProductVersionName, and FL_CreditActionCode.
# * Combine categories with frequency < threshold together
# * Obtain proportion of response variable for each level of the categorical variable
# * Obtain proportion of target = "Yes" and select smoothing factor
# * Compute smoothed logeit for each level of the categorical variable

# Let's look at the distribution of StateGroup:

# In[267]:

data.groupby('StateGroup')['StateGroup'].agg({'Frequency':'count'})


# Combiine all categories with less than 25 observations together into a category called 'Other'

# In[268]:

state_group_map = {'LargeGA' : 'Other' , 'LargeIN' : 'Other', 'LargeMI' : 'Other',
                'LargeOH' : 'Other'}
data['StateGroup'].replace(state_group_map, inplace = True)


# We can check the counts again to see that they are more evenly distributed:

# In[269]:

data.groupby('StateGroup')['StateGroup'].agg({'Frequency':'count'})


# Repeat the provess for ProductVersionName.

# In[270]:

data.groupby('ProductVersionName')['ProductVersionName'].agg({'Frequency':'count'})


# In[271]:

product_version_map = {'10 Tier' : 'Other', 'CASummit' : 'Other',
                       'Imperial' : 'Other', 'Rad2.1' : 'Other'}
data['ProductVersionName'].replace(product_version_map, inplace = True)


# The FL_CreditActionCode is distributed evenly enough that we do not need to cluster the low frequency categories.

# Let's view the proportion of the response variable, 'firstterm_survival', for each level of 'StateGroup'

# In[272]:

StateGroupSmoothingDf = data.groupby(['StateGroup','firstterm_survival'], as_index = False)['StateGroup'] .agg({'Frequency':'count'}).pivot('StateGroup','firstterm_survival').fillna(0)['Frequency']
StateGroupSmoothingDf


# Now let's calculate the smoothed logit for each category of StateGroup

# In[273]:

StateGroupSmoothingDf['Frequency'] = StateGroupSmoothingDf[0] + StateGroupSmoothingDf[1]
StateGroupSmoothingDf['Churn_Frequency'] = StateGroupSmoothingDf[1]
StateGroupSmoothingDf['Churn_Rate'] = StateGroupSmoothingDf['Churn_Frequency']/StateGroupSmoothingDf['Frequency']
StateGroupLength = len(data['StateGroup'])
StateGroupYes = len(data['StateGroup'][data['firstterm_survival'] == 1])
StateGroupNo = len(data['StateGroup'][data['firstterm_survival'] == 0])
Population_Proportion_Yes = float(StateGroupYes)/float(StateGroupLength)
Population_Proportion_No = float(StateGroupNo)/float(StateGroupLength)
SmoothingFactor = 2
StateGroupSmoothingDf['Logit_StateGroup'] = np.log((StateGroupSmoothingDf[1] + Population_Proportion_Yes*SmoothingFactor)/
                                       (StateGroupSmoothingDf[0] + Population_Proportion_No*SmoothingFactor))
StateGroupSmoothingDf


# Repeat logit smoothing for ProductVersionName and FL_CreditActionCode

# In[274]:

ProductSmoothingDf = data.groupby(['ProductVersionName','firstterm_survival'], as_index = False)['ProductVersionName'] .agg({'Frequency':'count'}).pivot('ProductVersionName','firstterm_survival').fillna(0)['Frequency']
ProductSmoothingDf['Frequency'] = ProductSmoothingDf[0] + ProductSmoothingDf[1]
ProductSmoothingDf['Churn_Frequency'] = ProductSmoothingDf[1]
ProductSmoothingDf['Churn_Rate'] = ProductSmoothingDf['Churn_Frequency']/ProductSmoothingDf['Frequency']
ProductSmoothingDf


# In[275]:

ProductLength = len(data['ProductVersionName'])
ProductYes = len(data['ProductVersionName'][data['firstterm_survival'] == 1])
ProductNo = len(data['ProductVersionName'][data['firstterm_survival'] == 0])
Population_Proportion_Yes = float(ProductYes)/float(ProductLength)
Population_Proportion_No = float(ProductNo)/float(ProductLength)
SmoothingFactor = 2

ProductSmoothingDf['Logit_ProductVersionName'] = np.log((ProductSmoothingDf[1] + Population_Proportion_Yes*SmoothingFactor)/
                                     (ProductSmoothingDf[0] + Population_Proportion_No*SmoothingFactor))
ProductSmoothingDf


# In[276]:

CreditActionSmoothingDf = data.groupby(['FL_CreditActionCode','firstterm_survival'], as_index = False)['FL_CreditActionCode'] .agg({'Frequency':'count'}).pivot('FL_CreditActionCode','firstterm_survival').fillna(0)['Frequency']
CreditActionSmoothingDf['Frequency'] = CreditActionSmoothingDf[0] + CreditActionSmoothingDf[1]
CreditActionSmoothingDf['Churn_Frequency'] = CreditActionSmoothingDf[1]
CreditActionSmoothingDf['Churn_Rate'] = CreditActionSmoothingDf['Churn_Frequency']/CreditActionSmoothingDf['Frequency']
CreditActionSmoothingDf


# In[277]:

CreditActionLength = len(data['FL_CreditActionCode'])
CreditActionYes = len(data['FL_CreditActionCode'][data['firstterm_survival'] == 1])
CreditActionNo = len(data['FL_CreditActionCode'][data['firstterm_survival'] == 0])
Population_Proportion_Yes = float(CreditActionYes)/float(CreditActionLength)
Population_Proportion_No = float(CreditActionNo)/float(CreditActionLength)
SmoothingFactor = 2
CreditActionSmoothingDf['Logit_FL_CreditActionCode'] = np.log((CreditActionSmoothingDf[1] + Population_Proportion_Yes*SmoothingFactor)/
                                     (CreditActionSmoothingDf[0] + Population_Proportion_No*SmoothingFactor))
CreditActionSmoothingDf


# Merge dataframes to assign smoothed logit to original dataset

# In[278]:

data = data.join(StateGroupSmoothingDf['Logit_StateGroup'], 
                 on='StateGroup', how='inner')
data = data.join(ProductSmoothingDf['Logit_ProductVersionName'], 
                 on='ProductVersionName', how='inner')
data = data.join(CreditActionSmoothingDf['Logit_FL_CreditActionCode'], 
                 on='FL_CreditActionCode', how='inner')


# Create a Cardinality Table for all categorical variables at this stage.

# In[282]:

catCols = ['affin_grp', 'AIRB_disc', 'BIGroupNum', 'DistributionChannelName',
           'firstterm_survival', 'FL_CreditActionCode', 'HomeOwner_disc',
           'HomeownerInd', 'lonly', 'max22', 'max30', 'MI_CP_limit', 'MI_PIP_limit', 
           'move', 'Multicar_disc', 'paymentmethodlong', 'PaymentMethodName',
           'PIF_disc', 'pointed', 'PreferredMailDocumentsCode', 'premcat', 
           'PriorCarrierTypeCode', 'ProductVersionName', 'season', 'StateGroup', 
           'StateRegion', 'TermInMonthsNum', 'uwtiergroup', 'year', 'Yearmonth']
for col in catCols:
    print data[col].value_counts()


# # Step 3: Numerical Variables Preparation

# Create missing value indicator for variable MP_LIMIT

# In[283]:

data['MI_MP_Limit'] = data['MP_Limit'].fillna(1)
data.loc[data['MI_MP_Limit'] != 1] = 0


# Use Median imputation on the following variables: CL_Limit, PD_Limit, MP_LIMIT, CREDITSCORENUM, INSURANCEEXPERIENCEDAYSNUM, PRECREDITTIERNUM, PRIORSWITCHESCOUNT, RATEMANUALNUM, and UWTIERNUM

# In[284]:

data['CL_Limit'].fillna(data['CL_Limit'].median(), inplace=True)
data['PD_Limit'].fillna(data['PD_Limit'].median(), inplace=True)
data['MP_Limit'].fillna(data['MP_Limit'].median(), inplace=True)
data['CreditScoreNum'].fillna(data['CreditScoreNum'].median(), inplace=True)
data['InsuranceExperienceDaysNum'].fillna(data['InsuranceExperienceDaysNum'].median(), inplace=True)
data['PreCreditTierNum'].fillna(data['PreCreditTierNum'].median(), inplace=True)
data['PriorSwitchesCount'].fillna(data['PriorSwitchesCount'].median(), inplace=True)
data['RateManualNum'].fillna(data['RateManualNum'].median(), inplace=True)
data['uwtiergroup'].fillna(data['uwtiergroup'].median(), inplace=True)


# Drop N_PAYMENT due to high correlation with PAYINSTALLMENTSNUM 

# In[285]:

data = data.drop('N_Payment', axis=1)


# Restrict the variable PNIAGE in the interval [16, 100]

# In[286]:

data = data[data['PNIAge'].between(16,100, inclusive = True)]


# Split the variable VEHXDRV to two variables N_Vehicles and N_Drivers.  The first digit to represent the number of vehicle and the second digit to represent the number of drivers.

# In[287]:

data['N_Vehicles'] = data['vehxdrv'].astype(str).str.slice(0,1,1)
data['N_Drivers'] = data['vehxdrv'].astype(str).str.slice(1,2,1)


# Perform rank transformation on the following variables RATEMANUALNUM N_MOD_BI_LIMIT FIRMCODE AGEGE75_LT30_LT21_POINTED PD_LIMIT MP_LIMIT MAXVEHVALUE PIP_LIMIT NEXTPREMCH

# In[288]:

data['Rank_RateManualNum'] = data['RateManualNum'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_BI_Limit'] = data['BI_Limit'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_FirmCode'] = data['FirmCode'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_agege75_lt30_lt21_pointed'] = data['agege75_lt30_lt21_pointed'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_PD_Limit'] = data['PD_Limit'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_MP_Limit'] = data['MP_Limit'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_maxvehvalue'] = data['maxvehvalue'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_PIP_Limit'] = data['PIP_Limit'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_nextpremch'] = data['nextpremch'].rank(axis = 0, method='average', na_option='bottom', ascending=True)


# Perform log transformation on the following variables: UWTIERNUM NONCANCELENDMTS INCURRED_LOSS. 

# In[291]:

data['Log_uwtiernum'] = np.log(data['uwtiergroup'])
data['Log_noncancelendmts'] = np.log(data['noncancelendmts'])
data['Log_incurred_loss'] = np.log(data['incurred_loss'])


# Perform rank transformation of the following three variables: DAYSLAPSENUM LOGIT_PRODUCTVERSIONNAME LASTENDMT

# In[293]:

data['Rank_DaysLapseNum'] = data['DaysLapseNum'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_Logit_ProductVersionName'] = data['Logit_ProductVersionName'].rank(axis = 0, method='average', na_option='bottom', ascending=True)
data['Rank_lastendmt'] = data['lastendmt'].rank(axis = 0, method='average', na_option='bottom', ascending=True)


# Perform square transformation of the following variables: LOGIT_FL_CREDITACTIONCODE and LOGIT_STATEGROUP

# In[294]:

data['Square_Logit_FL_CreditActionCode']= data['Logit_FL_CreditActionCode']**2
data['Square_Logit_StateGroup']= data['Logit_StateGroup']**2


# Perform power transformation of the following variables: 
# * TOTAL_FEE: Power 0.5 
# * TOTAL_ANN_PREM: Power 0.3125 
# * CL_LIMIT: Power 0.25 
# * NPCHCAT: Power 0.125 
# * INSURANCEEXPERIENCEDAYSNUM: Power 0.375 
# * CP_LIMIT: Power 0.4325

# In[296]:

data['Power_total_fee']= data['total_fee']**(.5)
data['Power_total_ann_prem']= data['total_ann_prem']**(.3125)
data['Power_CL_Limit']= data['CL_Limit']**(.25)
data['Power_npchcat']= data['npchcat']**(.125)
data['Power_InsuranceExperienceDaysNum']= data['InsuranceExperienceDaysNum']**(.375)
data['Power_CP_Limit']= data['CP_Limit']**(.4325)


# In[297]:

data


# In[ ]:



