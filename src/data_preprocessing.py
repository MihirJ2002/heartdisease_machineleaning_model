from sklearn.preprocessing import MinMaxScaler,RobustScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE



def preprocessing(df):

    df.drop_duplicates()

    df.drop(columns = ['patient_id','wearable_owner','family_history','exercise_induced_angina'],inplace = True)
    numerical_data = df.select_dtypes(exclude = 'object')
    categorical_data = df.select_dtypes(include = 'object')

    le = LabelEncoder()

    for i in categorical_data.columns:
        df[i] = le.fit_transform(df[i])

    X = df.drop(columns = ['has_heart_disease'])
    y = df['has_heart_disease']

    X_train,X_test,y_train,y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 1)
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    X_train,y_train = sm.fit_resample(X_train,y_train)

    return X_train, X_test, y_train, y_test