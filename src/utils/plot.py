#绘制类别变量中各子变量与target的柱状图和折现图
def plotbar(df,var,target):
    a=df.groupby([var])[target].mean().reset_index()
    b=df.groupby([var])[target].count().reset_index()
    b=b.merge(a,on=var,how="left").rename(columns={target+"_x":"counts",target+"_y":"rate"})
    summary=b
    summary=summary.sort_index(by="rate",ascending=False)
    summary["占比"]=summary["counts"]/summary["counts"].sum()
    summary["累计"]=summary["占比"].cumsum()#     summary=summary.sort_values(by="rate")
    name=str(b[var].tolist())
    ticks=range(len(b))
    y=b["counts"]
    #y1=aa["target"]
    fig = plt.figure(figsize=(10,6)
                              ,facecolor='#ffffcc',edgecolor='#ffffcc')#图片大小
    plt.grid(axis='y',color='#8f8f8f',linestyle='--', linewidth=1 )
    plt.xticks(rotation=60,fontsize=15)
    plt.xlabel('xlabel', fontsize=30)
    plt.yticks(fontsize=15)
    ax1 = fig.add_subplot(111)#添加第一副图
    ax2 = ax1.twinx()#共享x轴，这句很关键！
    plt.yticks(fontsize=15)
    plt.ylabel('ylabel', fontsize=30)
    sns.pointplot(b[var],b["rate"],marker="o",ax=ax1)
    sns.barplot(b[var],b["counts"],ax=ax2,alpha=0.5)
    for a,b in zip(ticks,y):
        plt.text(a-0.3,b ,str(b),fontsize=20)
    ax1.set_ylabel('rate', size=18)
    ax2.set_ylabel('counts', size=18)
    return summary
#示例
plotbar(df,'gender,'target')

def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    """
    横向き棒グラフ作成関数
    df:  
    col: 
    title: 
    color: 
    w=None: 
    h=None: 
    lm=0: 
    limit=100: 
    return_trace=False: 
    rev=False: 
    xlb = False:
    """
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    """
    グループ化棒グラフを表示
    col: 表示する列
    title: 図のタイトル
    """
    df1 = app_train[app_train["TARGET"] == 1]
    df0 = app_train[app_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()

    trace1 = go.Bar(x=a1.index, y=a1.values, name='Target : 1', marker=dict(color="#44ff54"))
    trace2 = go.Bar(x=b1.index, y=b1.values, name='Target : 0', marker=dict(color="#ff4444"))

    data = [trace1, trace2]
    layout = go.Layout(barmode='group', height=300, title = title)

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')
    
def exploreCat(col):
    t = application[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)

#the realation between the categorical column and the target 
def catAndTrgt(col):
    tr0 = bar_hor(application, col, "Distribution of "+col ,"#f975ae", w=700, lm=100, return_trace= True)
    tr1, tr2 = gp(col, 'Distribution of Target with ' + col)

    fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = [col +" Distribution" , "% Rpyment difficulty by "+col ,"% of otherCases by "+col])
    fig.append_trace(tr0, 1, 1);
    fig.append_trace(tr1, 1, 2);
    fig.append_trace(tr2, 1, 3);
    fig['layout'].update(height=350, showlegend=False, margin=dict(l=50));
    iplot(fig);



# 11. 绘制学习曲线，以确定模型的状况是否过拟合和欠拟合
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,
                        n_jobs=None,train_sizes=np.linspace(.1,1.0,10)):
    """
    生成训练和测试的学习曲线图
    参数:
    ---------------------
    estimator: object type
    title: string
           图表的标题
    X: 类数组，形状(n_samples, n_features)
       训练向量，其中n_samples为样本个数,n_features是特性的数量。
    y: 类数组，形状(n_samples)或(n_samples, n_features)，可选目标相对于X进行分类或回归;
    ylim:元组，形状(ymin, ymax)，可选定义绘制的最小和最大y值。
    cv:int，交叉验证生成器或可迭代的，可选的确定交叉验证拆分策略。
        cv的可能输入是:
            -无，使用默认的3倍交叉验证，
            -整数，指定折叠的次数。
    n_jobs:int或None，可选(默认=None) 并行运行的作业数。'None'的意思是1。
           “-1”表示使用所有处理器。
    train_sizes：类数组，形状(n_ticks，)， dtype float或int
                相对或绝对数量的训练例子，将用于生成学习曲线。如果dtype是float，则将其视为
                训练集的最大大小的分数(这是确定的)，即它必须在(0,1)范围内。
                否则，它被解释为训练集的绝对大小。
                注意，为了分类，样本的数量通常必须足够大，可以包含每个类的至少一个示例。
                (默认:np.linspace(0.1, 1.0, 5))
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#示例：
# from sklearn.datasets import  make_gaussian_quantiles
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# import numpy as np
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# # ##########################
# # 生成2维正态分布，生成的数据按分位数分为两类，50个样本特征，5000个样本数据
# X,y = make_gaussian_quantiles(cov=2.0,n_samples=5000,n_features=50,
#                               n_classes=2,random_state=1)
# # 设置一百折交叉验证参数，数据集分层越多，交叉最优模型越接近原模型
# cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=1)
# # 分别画出CART分类决策树和AdaBoost分类决策树的学习曲线
# estimatorCart = DecisionTreeClassifier(max_depth=1)
# estimatorBoost = AdaBoostClassifier(base_estimator=estimatorCart,
#                                     n_estimators=270)
# # 画CART决策树和AdaBoost的学习曲线
# estimatorTuple = (estimatorCart,estimatorBoost)
# titleTuple =("decision learning curve","adaBoost learning curve")
# title = "decision learning curve"
# for i in range(2):
#     estimator = estimatorTuple[i]
#     title = titleTuple[i]
#     plot_learning_curve(estimator,title, X, y, cv=cv)
# plt.show()
    

