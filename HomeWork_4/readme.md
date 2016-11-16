This is for Homework4 for Ada. 
----------------------
Link to cross validation
----------------------

http://scikit-learn.org/stable/modules/cross_validation.html



-------------------------
This is useful for ROC plot
-----------------------
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    X_train2 = X_train[:, [4, 14]]
    cv = StratifiedKFold(y_train,
                             n_folds=3,
                             random_state=1)
    fig = plt.figure(figsize=(7, 5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train, test) in enumerate(cv):
            probas = pipe_lr.fit(X_train2[train],
        y_train[train]).predict_proba(X_train2[test])
            fpr, tpr, thresholds = roc_curve(y_train[test],probas[:, 1],pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)                                 
            mean_tpr[0] = 0.0                                 
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,tpt,lw=1,label='ROC fold %d (area = %0.2f)', % (i+1, roc_auc))

    plt.plot([0, 1],[0, 1],   linestyle='--', color=(0.6, 0.6, 0.6),label='random guessing')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1],[0, 1, 1],lw=2,linestyle=':', color='black',label='perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc="lower right")
    plt.show()        



-------------------
This I found good for ROC plot too.
-----------------
    cv = StratifiedKFold(n_splits=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0
    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

