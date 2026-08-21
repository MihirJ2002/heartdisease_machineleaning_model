from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_model(X_train, y_train, X_test, y_test):
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    
    # Train the model
    rf.fit(X_train, y_train)  # seen data
    
    # Make predictions
    y_pred = rf.predict(X_test) # Unseen data
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report and confusion matrix
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return rf