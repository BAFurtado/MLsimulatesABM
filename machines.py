from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


def run_random_forest_cross(x, y):
    x, y = shuffle(x, y)
    m1 = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=True)
    accuracies = cross_val_score(m1, x, y['target'], cv=2)
    print('All accuracies RF Automatic')
    print(accuracies)
    print('Average accuracy Random Forest Automatic resampling {:.4f}'.format(sum(accuracies)/len(accuracies)))
    return m1


def run_random_forest_split(x, y):
    x, y = shuffle(x, y)
    m1 = RandomForestClassifier(n_estimators=1000, criterion='entropy', bootstrap=True)
    train_size = int(len(x) * .7)
    m1.fit(x[train_size:], y['target'][train_size:])
    accuracy = m1.score(x[:train_size:], y['target'][:train_size])
    print('Accuracy Random Forest Manual data-splitting {:.4f}'.format(accuracy))
    return m1


def predict_random_forest_cross(model, data):
    return model.predict(data)
