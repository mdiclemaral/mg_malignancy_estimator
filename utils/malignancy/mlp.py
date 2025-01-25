import pandas as pd


class MLP:
    def test_xgboost_model(self, model_pickle, df):
        df.dropna(inplace=True)
        model, scaler, features = model_pickle
        
        # X_test = df.drop(columns=['sample_id'])  # ??
        X_selected = df[features]
        # print(X_selected.columns == df.columns)
        X_test = scaler.transform(X_selected)
        
        # print(len(X_test.columns))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        # here, X_selected is returned instead of X_test because it is numpy.ndarray
        return X_selected, y_pred, y_proba[:, 1]

    def save_predictions(self, df, X_test, y_pred, y_proba):
        # Map the test indices back to the external_ids
        test_indices = X_test.index
        results = df.loc[test_indices, ['sample_id']].copy()
        results['prediction'] = y_pred
        results['probability'] = y_proba

        results = results.set_index('sample_id').T.to_dict('list')
        return results

    def test_model(self, model_pickle, dataset):
        X_test, y_pred, y_proba = self.test_xgboost_model(model_pickle, dataset)
        results = self.save_predictions(dataset, X_test, y_pred, y_proba)
        print("AAAA:", y_proba)
        print("")
        return results
        