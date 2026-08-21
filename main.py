from src.data_ingestion import data_loader
from src.data_preprocessing import preprocessing
from src.model_building import build_model

def main():

    #Step1 : Data_Ingestion
    df = data_loader()
    print(df.shape)

    #Step2: data_preprocessing
    X_train, X_test, y_train, y_test = preprocessing(df)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #Step3 : model_building
    model = build_model(X_train, y_train, X_test, y_test)


main()