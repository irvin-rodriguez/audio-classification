from model_utils import plot_confusion_matrix
import joblib

import torch
from FNN_executioner import Net
from CNN_executioner import CNN

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def main():
    # load in train data for SVM and FNN 
    X, y = joblib.load('./data/processed/avg_mfcc_data.pkl')
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # load in train data for CNN
    X, y = joblib.load('./data/processed/cnn_mfcc_data.pkl')
    _, X_test_cnn, _, y_test_cnn = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # svm model (scalar included in model pipeline already)
    model_svm = joblib.load("./models/best_svm_model.pkl")

    y_pred_svm = model_svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

    # fnn model
    model_fnn = Net()  # create instance of Net model
    model_fnn.load_state_dict(torch.load("./models/best_fnn_model.pth"))
    model_fnn.eval()  # switch to eval mode 

    scalar_fnn = joblib.load("./models/fnn_scaler.pkl")  # import scalar used as well
    X_test_fnn_scaled = scalar_fnn.transform(X_test)
    X_test_fnn_tensor = torch.tensor(X_test_fnn_scaled, dtype=torch.float32)

    with torch.no_grad():
        outputs_fnn = model_fnn(X_test_fnn_tensor)
        _, y_pred_fnn = torch.max(outputs_fnn, 1)
    print("FNN Accuracy:", accuracy_score(y_test, y_pred_fnn.numpy()))
    print("FNN Classification Report:\n", classification_report(y_test, y_pred_fnn.numpy()))

    # cnn model
    model_cnn = CNN()
    model_cnn.load_state_dict(torch.load("./models/best_cnn_model.pth"))
    model_cnn.eval()

    X_test_cnn_tensor = torch.tensor(X_test_cnn, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs_cnn = model_cnn(X_test_cnn_tensor)
        _, y_pred_cnn = torch.max(outputs_cnn, 1)
    print("CNN Accuracy:", accuracy_score(y_test_cnn, y_pred_cnn.numpy()))
    print("CNN Classification Report:\n", classification_report(y_test_cnn, y_pred_cnn.numpy()))

    # confusion matrices (normalized)
    plot_confusion_matrix(y_test, y_pred_svm, "svm")
    plot_confusion_matrix(y_test, y_pred_fnn.numpy(), "fnn")
    plot_confusion_matrix(y_test, y_pred_cnn.numpy(), "cnn")

    return 0

if __name__ == "__main__":
    main()