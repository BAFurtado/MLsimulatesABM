from sklearn.metrics import confusion_matrix


def print_model(x, m1, train_size):
    yhat = m1.predict(x[train_size:])
    cm = confusion_matrix(y[train_size:], yhat)
    print('Confusion matrix for RF Manual')
    print(cm)
    print('Features importance: ')
    print('')
    out = dict()
    for i in range(len(m1.feature_importances_)):
        if m1.feature_importances_.item(i) > 0:
            out[x.columns[i]] = m1.feature_importances_.item(i)
    for w in sorted(out, key=out.get, reverse=True):
        print('{}: {:.4f}'.format(w, out[w]))
