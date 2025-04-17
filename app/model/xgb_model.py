import pickle
import pandas as pd



class Model:
    # model_path = "https://raw.githubusercontent.com/DerrickVericho/Model-Deployment/master/app/model/onehot_encoder.pkl"
    # preprocessor_path = "https://raw.githubusercontent.com/DerrickVericho/Model-Deployment/master/app/model/xgb_model.pkl"
    
    model_path = "model/xgb_model.pkl"
    preprocessor_path = "model/onehot_encoder.pkl"

    def __init__(self, model_path=model_path, preprocessor_path=preprocessor_path) -> None: # model_path: str, preprocessor_path: str):
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
            
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
    def read_data(self, path):
        return pd.read_csv(path, encoding='ISO-8859-1')
            
    def preprocess_input(self, input_dict: dict) -> pd.DataFrame:
        """
        Ubah input user (dict) jadi dataframe lalu transform pakai preprocessor
        """
        df = pd.DataFrame([input_dict])
        return self.preprocessor.transform(df)

    def predict(self, input_dict: dict) -> int:
        """
        Hasil prediksi akhir, return 0 / 1
        """
        X_transformed = self.preprocess_input(input_dict)
        return self.model.predict(X_transformed)[0]

    def predict_proba(self, input_dict: dict) -> list:
        """
        Probabilitas prediksi [prob_0, prob_1]
        """
        X_transformed = self.preprocess_input(input_dict)
        return self.model.predict_proba(X_transformed)[0]
        
        