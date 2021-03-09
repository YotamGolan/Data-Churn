import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense



def neural_network():
    EPOCHS = 40
    intake_data = pd.read_csv('Customer_Churn_processed.csv')

    X = intake_data.drop(columns=['LEAVE'])
    X = StandardScaler().fit_transform(X)
    Y = intake_data['LEAVE']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=True)

    model = Sequential()
    model.add(Dense(100, input_dim=11, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=EPOCHS)

    _ , final_accuracy = model.evaluate(X_test, Y_test)
    print(final_accuracy)

    # plot
    plt.plot(history.history['accuracy'])
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

def decision_tree():
    intake_data = pd.read_csv('Customer_Churn_processed.csv')

    X = intake_data.drop(columns=['LEAVE'])
    Y = intake_data['LEAVE']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=True)

    Tree = DecisionTreeClassifier(max_depth=8)
    Tree = Tree.fit(X_train, Y_train)

    Y_pred = Tree.predict(X_test)

    print(accuracy_score(Y_test, Y_pred))

#neural_network()
decision_tree()
