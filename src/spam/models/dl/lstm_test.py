# LSTM Testing Script

def test_lstm(model, X_test):
    return (model.predict(X_test) > 0.5).astype("int32")