import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    
    model = load_model('GBC')
    predictions = predict_model(model, df)
    predictions.rename({'prediction_label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
    return predictions[['Churn_prediction','prediction_score']]

def correct_features(df):
    df['PhoneService'] = df['PhoneService'].replace({'No': 0, 'Yes': 1}).infer_objects(copy=False)
    df['Contract'] = df['Contract'].replace({'Month-to-month': 0, 'One year': 1, 'Two year' : 2}).infer_objects(copy=False)
    df['PaymentMethod'] = df['PaymentMethod'].replace({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)' : 2,
                                                               'Credit card (automatic)' : 3}).infer_objects(copy=False)
    df['tenure'] = df['tenure'].replace({0 :1}).infer_objects(copy=False)
    df['TotalCharge_MonthlyCharge_ratio'] = df['TotalCharges'] / df['MonthlyCharges']
    df['TotalCharge_tenure_ratio'] = df['TotalCharges'] / df['tenure']
    df['Automatic_payment'] = df['PaymentMethod'].apply(lambda x: 1 if x in [2, 3] else 0)

    return df

if __name__ == "__main__":
    df = load_data('new_churn_data_unmodified.csv')
    churn_df = correct_features(df)
    predictions = make_predictions(churn_df)
    print('predictions:')
    print(predictions)
