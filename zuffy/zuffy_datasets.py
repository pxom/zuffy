import numpy as np
import pandas as pd
from zuffy.fuzzy_transformer import FuzzyTransformer, convert_to_numeric
DATASET_FOLDER = '../../datasets/'

def prep_data_mushrooms():
    # fetch dataset 
    my_data = pd.read_csv(DATASET_FOLDER + 'mushrooms.csv', sep=',', header=0, skiprows=0)
    print('mush loaded')

    dataset_name   = 'mushroom'
    target_name    = 'class'
    _, my_data = convert_to_numeric(my_data, target_name)
    target_class_names = ['poisonous', 'edible']

    X = my_data.iloc[:,1:]
    y = my_data[target_name]

    non_fuzzy = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat']

    ds_zc_params = {'parsimony_coefficient':0.00025}
    ds_it_params = {'n_iter':15}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_iris():
    from sklearn.datasets import load_iris

    iris           = load_iris()
    dataset_name   = 'iris'
    target_name    = 'target'
    my_data        = pd.DataFrame(data=np.c_[iris['data'],iris[target_name]],columns=iris['feature_names']+[target_name])
    target_class_names = list(iris.target_names)
    X              = my_data.iloc[:,0:-1]
    y              = my_data.iloc[:,-1]
    non_fuzzy      = []

    ds_zc_params = {'parsimony_coefficient':0.0004,'generations':15}
    ds_it_params = {'n_iter':5}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_penguin():
    dataset_name = 'penguin'
    my_data = pd.read_csv(DATASET_FOLDER + 'penguins_size.csv', sep=',', header=0, skiprows=0)
    target_name = 'species'
    non_fuzzy = ['sex','island']

    # We remove all rows that contain any null values
    for f in my_data:
        if my_data[f].isna().sum() > 0:
            print(f"Feature {f:<16} has {my_data[f].isna().sum(): 6.0f} null values so we drop those rows.")

    my_data.dropna(how='any', inplace=True)

    # Drop rows where sex is unknown
    my_data = my_data[ my_data['sex'].isin(['MALE','FEMALE'])]

    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient':0.0003}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_australian():
    dataset_name = 'australian'
    non_fuzzy = ['d0','d3','d4','d5','d7','d8','d10','d11']
    my_data = pd.read_csv(DATASET_FOLDER + 'australian.dat', sep=' ', header=0, skiprows=0)
    target_name = 'output'
    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient': 0.0001,'population_size':1000,'generations':15}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_haberman():
    dataset_name = 'haberman'
    my_data = pd.read_csv(DATASET_FOLDER + 'haberman.csv', sep=',', header=None, skiprows=0)
    features = ['age','operation_year','positive_auxillary_nodes']
    target_name = 'survival_status'
    combined = features + [target_name]
    my_data.columns = combined
    target_class_names, my_data = convert_to_numeric(my_data, target_name)
    non_fuzzy = []

    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient': 0.0001,'population_size':1000,'generations':15,'tournament_size':25}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_banknote():
    dataset_name = 'banknote'
    my_data = pd.read_csv(DATASET_FOLDER + 'banknote_Train.csv', sep=' ', header=0, skiprows=0)
    target_name = 'y'

    non_fuzzy = []
    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    ds_zc_params = {'parsimony_coefficient': 0.0009,'population_size':1300,'generations':30,'tournament_size':80}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, list(target_class_names), ds_zc_params, ds_it_params

def prep_data_pima(): # diabetes
    dataset_name = 'pima'
    my_data = pd.read_csv(DATASET_FOLDER + 'diabetes.csv', sep=',') # , header=0, skiprows=1
    my_data.index = my_data.index.astype(int)
    target_name = 'Outcome'
    _, my_data = convert_to_numeric(my_data, target_name)
    target_class_names = ['No Diabetes','Has Diabetes']
    non_fuzzy = []

    X = my_data.iloc[:,0:-1]
    y = my_data.iloc[:,-1]

    ds_zc_params = {'parsimony_coefficient': 0.0003,'population_size':1500,'generations':15,'tournament_size':40}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_wine():
    from sklearn.datasets import load_wine
    wine = load_wine()
    dataset_name = 'wine'
    target_name = 'target'
    my_data=pd.DataFrame(data=np.c_[wine['data'],wine[target_name]],columns=wine['feature_names']+[target_name])
 
    non_fuzzy = []
    _, my_data = convert_to_numeric(my_data, target_name)
    target_class_names = list(wine.target_names)
    X = my_data.iloc[:,0:-1]
    y = my_data.iloc[:,-1]

    ds_zc_params = {'parsimony_coefficient': 0.005,'population_size':2500,'generations':20,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_breast():
    from sklearn.datasets import load_breast_cancer
    breast = load_breast_cancer(as_frame=True)
    dataset_name = 'breast'
    target_name = 'target'
    my_data=pd.DataFrame(data=np.c_[breast['data'],breast[target_name]],columns=list(breast['feature_names'])+[target_name])
 
    non_fuzzy = []
    _, my_data = convert_to_numeric(my_data, target_name)
    target_class_names = list(breast.target_names)
    X = my_data.iloc[:,0:-1]
    y = my_data.iloc[:,-1]

    ds_zc_params = {'parsimony_coefficient': 0.005,'population_size':2500,'generations':20,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_cleveland():
    dataset_name = 'cleveland'
    my_data = pd.read_csv(DATASET_FOLDER + 'processed.cleveland.data', sep=',', header=None, skiprows=0)
    my_data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']

    # need to drop rows with ca=?
    i = my_data[((my_data.ca == '?') | (my_data.thal == '?'))].index
    my_data.drop(i, inplace=True)

    target_name = 'class'
    # we want to use this data in a binary classification context so map all values >0 (ie. those that indicate disease) to 1
    my_data.loc[my_data[target_name]>1,target_name] = 1
    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    # convert all data to float
    my_data = my_data.astype(float)

    non_fuzzy = ['sex','cp','fbs','restecg','exang','slope','thal']
    # convert non fuzzy back to integers for neater display
    my_data[non_fuzzy] = my_data[non_fuzzy].astype(int)

    # do we actually want to normalise here? I don't think so and the results don't appear to be better
    features_to_normalise = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    my_slice = my_data.loc[:, features_to_normalise]
    my_slice = (my_slice - my_data.mean())/my_slice.std()
    my_data[features_to_normalise] = my_slice[features_to_normalise]
    
    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient': 0.001,'population_size':3000,'generations':25,'tournament_size':75}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_adult(): # UCI - can't find url.  note: using training dataset only for now
    dataset_name = 'adult'
    my_data = pd.read_csv(DATASET_FOLDER + 'adult.data', sep=',', header=None, skiprows=0)
    '''
    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
'''
    my_data.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex',
                       'capital-gain','capital-loss','hours-per-week','native-country','income']

    my_data.drop(['education-num'], axis=1, inplace=True) # correlated directly to education

    target_name = 'income'
    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    non_fuzzy =  ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient': 0.00005,'population_size':1000,'generations':50,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_bank_marketing():
    dataset_name = 'bank_marketing'
    non_fuzzy = ['job','marital','education','default','housing','loan','contact','poutcome'] 
    #age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed,class

    my_data = pd.read_csv(DATASET_FOLDER + 'bank-additional-full.csv', sep=',', header=0, skiprows=0)
    # Drop columns where feature is not useful
    my_data.drop(['month','day_of_week','duration'],inplace=True,axis=1)

    target_name = 'class'
    #target_classes, my_data = functions.convert_to_numeric(my_data, target_name)
    target_classes = []

    if 0:
        # ohe our non-numerics
        new_non_fuzzy = []
        for f in non_fuzzy:
            # one hot these
            print(f"One hot encoding {f:<16}")
            ohe = pd.get_dummies(my_data[f]).astype(int)
            #df["somecolumn"] = df["somecolumn"].astype(int)
            pfx_cols = [str(f) + ": " + str(item) for item in list(ohe.columns)]
            ohe.columns = pfx_cols
            new_non_fuzzy.extend(pfx_cols)
            my_data = pd.concat([my_data, ohe], axis=1)
            my_data = my_data.drop(f, axis=1)

    target_class_names, my_data = convert_to_numeric(my_data, target_name)
    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient': 0.00005,'population_size':3000,'generations':15,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_german_credit():
    dataset_name = 'german_credit'
    non_fuzzy = ['d1', 'd3', 'd4', 'd6', 'd7', 'd9', 'd10', 'd12', 'd14', 'd15', 'd17', 'd19', 'd20']

    my_data = pd.read_csv(DATASET_FOLDER + 'german.data', sep=',', header=0, skiprows=0)

    if 0:
        # Map attribute values to something meaningful
        mapping = {
            'A11':'< 0DM', 'A12':'>= 0DM', 'A13':'>= 200DM',
            'A141':'bank', 'A142':'stores', 'A143':'none', 
            }
        for m in mapping:
            for c in my_data.columns:
                my_data.loc[my_data[c]==m, c] = mapping[m]

    target_name = 'class'
    target_classes = []
    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    if 0:
        # ohe our non-numerics
        new_non_fuzzy = []
        for f in non_fuzzy:
            # one hot these
            print(f"One hot encoding {f:<16}")
            ohe = pd.get_dummies(my_data[f]).astype(int)
            pfx_cols = [str(f) + ": " + str(item) for item in list(ohe.columns)]
            ohe.columns = pfx_cols
            new_non_fuzzy.extend(pfx_cols)
            my_data = pd.concat([my_data, ohe], axis=1)
            my_data = my_data.drop(f, axis=1)

    y = my_data[target_name]
    X = my_data.drop(target_name, axis=1)

    #fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X, new_non_fuzzy, info=False, tags=['low', 'med', 'high'])

    ds_zc_params = {'parsimony_coefficient': 0.001,'population_size':4000,'generations':15,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_lupus():
    dataset_name    = 'lupus'
    X_raw           = pd.read_csv(DATASET_FOLDER + "lupus.csv", sep=",")
    y_raw           = pd.read_csv(DATASET_FOLDER + "lupus_labels.csv")
    my_data         = pd.concat([X_raw,y_raw],ignore_index=False,sort=False,axis=1)
    target_name     = 'class'
    target_classes  = []
    non_fuzzy   = []
    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    y               = my_data[target_name]
    X               = my_data.drop(target_name, axis=1)

    ds_zc_params = {'parsimony_coefficient': 0.0001,'population_size':2500,'generations':25,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_transfusion():
    dataset_name    = 'transfusion'
    X_raw           = pd.read_csv(DATASET_FOLDER + "transfusion.csv", sep=",")
    y_raw           = pd.read_csv(DATASET_FOLDER + "transfusion_labels.csv")
    my_data         = pd.concat([X_raw,y_raw],ignore_index=False,sort=False,axis=1)
    target_name     = 'class'
    target_classes  = []
    new_non_fuzzy   = []
    non_fuzzy   = []
    target_class_names, my_data = convert_to_numeric(my_data, target_name)

    y               = my_data[target_name]
    X               = my_data.drop(target_name, axis=1)

    #fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X, new_non_fuzzy, info=False, tags=['low', 'med', 'high'])

    ds_zc_params = {'parsimony_coefficient': 0.0001,'population_size':4000,'generations':15,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_recidivism():
    dataset_name    = 'recidivism'
    my_data         = pd.read_csv(DATASET_FOLDER + "compas-scores-two-years.csv", sep=",")

    cols_to_keep = ['age', 'age_cat', 'c_charge_degree', 'days_b_screening_arrest', 'decile_score', 'is_recid',
                    'priors_count', 'race', 'score_text', 'sex', ]
    
    my_data = my_data[cols_to_keep]

    '''
    my_data.drop([ 'id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date',
                    'c_arrest_date', 'r_case_number', 'r_offense_date', 'r_jail_in', 'r_jail_out', 'r_charge_degree', 'r_days_from_arrest',
                    'r_charge_desc', 'violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
                    'type_of_assessment', 'screening_date', 'v_type_of_assessment', 'v_screening_date', 'in_custody', 'out_custody',
                    'is_violent_recid', 'event', 'two_year_recid', 'start', 'end', 'juv_fel_count', 'juv_other_count', 'juv_misd_count',
                    'c_days_from_compas', 'c_charge_desc', 'decile_score.1', 'v_decile_score', 'v_score_text', 'priors_count.1' ],
                     inplace=True, axis=1)
    '''

    # Filter rows where 'days_b_screening_arrest' is between -30 and 30
    my_data = my_data[(my_data['days_b_screening_arrest'] >= -30) & (my_data['days_b_screening_arrest'] <= 30)]


    target_name     = 'is_recid'
    target_classes  = []
    non_fuzzy       = ['age_cat','c_charge_degree','race','score_text','sex']
    target_class_names, my_data = convert_to_numeric(my_data, target_name)
    target_class_names = ['No', 'Yes']

    y               = my_data[target_name]
    X               = my_data.drop(target_name, axis=1)

    #fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X, non_fuzzy, info=False, tags=['low', 'med', 'high'])

    ds_zc_params = {'parsimony_coefficient': 0.0003,'population_size':1500,'generations':30,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params

def prep_data_stroke():
    dataset_name = 'stroke'
    my_data      = pd.read_csv(DATASET_FOLDER + "healthcare-dataset-stroke-data.csv", sep=",") # from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

    # drop unique id
    my_data.drop(['id'], inplace=True, axis=1)

    # Filter so we just consider people aged 18+ (only 2 people under that age had a stroke)
    my_data = my_data[(my_data['age'] >= 18)]

    # Set BMI to average (30.4) where unknown (but should probably adjust for age)
    my_data.loc[my_data['bmi'].isna(), "bmi"] = my_data['bmi'].mean()

    #my_data['bmi'] = my_data[(my_data['bmi'] == 'N/A')]

    # lazy! just dropping them
    #my_data.dropna(subset=['bmi'], inplace=True)


    target_name     = 'stroke'
    target_classes  = []
    non_fuzzy       = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']
    _, my_data = convert_to_numeric(my_data, target_name)
    target_class_names = ['no', 'yes']
    y               = my_data[target_name]
    X               = my_data.drop(target_name, axis=1)

    #fuzzy_X, fuzzy_features_names = functions.fuzzify_data(X, non_fuzzy, info=False, tags=['low', 'med', 'high'])

    ds_zc_params = {'parsimony_coefficient': 0.000001,'population_size':1500,'generations':15,'tournament_size':50}
    ds_it_params = {}
    return dataset_name, X, y, non_fuzzy, target_class_names, ds_zc_params, ds_it_params
