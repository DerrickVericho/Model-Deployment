# Modul EDA
import pandas as pd

# Modul Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle

class BestXGB:
    
    model_path = "app/model/xgb_model.pkl"
    preprocessor_path = "app/model/onehot_encoder.pkl"
    
    def __init__(self, model_path=model_path, preprocessor_path=preprocessor_path):
        
        # Load Encoder
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
            
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        return df
    
    def change_dtype_int(self, df):
        df['person_age'] = df['person_age'].astype(int)
        df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype(int)
        return df

    def clean_data(self, df):
        df = df.replace("fe male", "female")
        df['loan_status'] = df['loan_status'].astype(int)
        return df

    def remove_row(self, df):
        df = df.drop(df[df['person_age'] > 70].index)
        df = df.drop(df[df['person_income'] > 800000].index)
        return df
    
    def handling_person_income(self, df):
        df['person_income_missing'] = df['person_income'].isna().astype(int)
        df['person_income'] = df.groupby('person_education')['person_income'].transform(lambda x: x.fillna(x.median()))
        return df
    
    def define_X_y(self, df):
        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]
        return X, y
    
    def label_encoding(self, X):
        X['person_gender'] = X['person_gender'].map({"male":0, "female":1})
        X['previous_loan_defaults_on_file'] = X['previous_loan_defaults_on_file'].map({"No":0, "Yes":1})
        return X
    
    def preprocess(self, X):
        return self.preprocessor.transform(X)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def predict(self, X):
        return self.model.predict(X)
        
    def eval_classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)
        
if __name__ == '__main__':
    trainer = BestXGB()
    df = trainer.load_data("Dataset_A_loan.csv")
    df = trainer.change_dtype_int(df)
    df = trainer.clean_data(df)
    df = trainer.remove_row(df)
    df = trainer.handling_person_income(df)
    X, y = trainer.define_X_y(df)
    X = trainer.label_encoding(X)
    X = trainer.preprocess(X)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    y_pred = trainer.predict(X_test)
    eval = trainer.eval_classification_report(y_test, y_pred)
    print(eval)