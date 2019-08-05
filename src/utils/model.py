def lr_predict(train_x,train_y,evl_x,evl_y):
    lr_model = LogisticRegression(C=0.1,class_weight='balanced')
    lr_model.fit(train_x,train_y)
    
    y_pred = lr_model.predict_proba(train_x)[:,1]
    fpr_lr,tpr_lr,_ = roc_curve(train_y,y_pred)
    train_ks = abs(fpr_lr - tpr_lr).max()
    print('train_ks : ',train_ks)
    
    y_pred = lr_model.predict_proba(evl_x)[:,1]
    fpr_lr,tpr_lr,_ = roc_curve(evl_y,y_pred)
    evl_ks = abs(fpr_lr - tpr_lr).max()
    print('evl_ks : ',evl_ks)
    return train_ks,evl_ks
    plt.plot(fpr_lr_train,tpr_lr_train,label = 'train LR')
    plt.plot(fpr_lr,tpr_lr,label = 'evl LR')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc = 'best')
    plt.show()

