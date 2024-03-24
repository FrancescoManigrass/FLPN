import os
from os.path import join

import matplotlib
import numpy as np
import pandas
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


folders = os.listdir("../features/")
folders = [f for f in folders if f.endswith(".csv")]
folders=[folders[0]]
for w in folders:
    #w="gzsl_seen_Experiment(NEW-1889)_AWA2_student_GZSL_id_0.csv"
    weight_path="/media/grains6lab2/f/prova/features/"+w
    base_path = "/media/grains6lab2/f/prova/data/wgr/code/APN-ZSL-master/data/AWA2/"
    dataframe=pandas.read_csv(weight_path).to_numpy()
    unique_class =[int(f) for f in  list(np.unique(dataframe[:,52].reshape(-1,1)))]


    prototype_df=dataframe[:,1:51].reshape(-1,50)
    predicted_labels=dataframe[:,51].reshape(-1,1)
    gt_labels=dataframe[:,52].reshape(-1,1)

    indexs=[]
    my_dict  = {}


    gt_labels_list=[f[0] for f in gt_labels]

    for i in range(len(gt_labels_list)):
        if gt_labels_list[i] in unique_class:
            if gt_labels_list[i] not in my_dict:
                my_dict[gt_labels_list[i]]=0
            if my_dict[gt_labels_list[i]] <500:
                my_dict[gt_labels_list[i]] += 1
                indexs.append(i)








    print("dfdfd")


    """
    # [f  for f in range(gt_labels.shape[0]) if gt_labels[f]==predicted_labels[f] ]#
    index =[ f for f in range(len(gt_labels)) if gt_labels[f] in unique_class]
    """
    prototype_df=prototype_df[indexs]
    predicted_labels=predicted_labels[indexs]
    gt_labels=gt_labels[indexs]



    att_splits = scipy.io.loadmat(base_path + "att_splits.mat") #FOR AWA2
    classes_names = att_splits['allclasses_names']
    classes_names = [classes_names[i][0][0] for i in range(classes_names.size)]
    for f in range(len(classes_names)):
        if "+" in classes_names[f]:
            v=classes_names[f].split("+")
            tmp=v[0]+"+"+v[1][0]+"."
            classes_names[f] = tmp

    #pca = PCA(n_components=50)
    #X_pca = pca.fit_transform(prototype_df)
    X_pca=prototype_df
    from sklearn import manifold
    """
    tsne = manifold.TSNE(n_components=2,learning_rate=0.0010,perplexity=60,
                         n_iter=4000,n_iter_without_progress=690,verbose=1,random_state=123).fit_transform(X_pca)

    """

    import plotly.express as px

    perplexity = np.arange(5, 500, 5)
    divergence = []

    for i in tqdm(perplexity):
        model = TSNE(n_components=2, init="pca", perplexity=i,n_jobs=8)
        reduced = model.fit_transform(X_pca)
        divergence.append(model.kl_divergence_)
    fig = px.line(x=perplexity, y=divergence, markers=True)
    fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
    fig.update_traces(line_color="red", line_width=1)
    fig.show()
    """
    tsne = manifold.TSNE(n_components=2,perplexity=10,
                         random_state=123,verbose=1).fit_transform(X_pca)
    """

    tsne = manifold.TSNE(n_components=2, learning_rate=0.0010, perplexity=60,
                         n_iter=4000, n_iter_without_progress=690, verbose=1, random_state=123).fit_transform(X_pca)

    df_subset = {}
    df_subset['tsne-2d-one'] = tsne[:, 0]
    df_subset['tsne-2d-two'] = tsne[:, 1]
    df_subset['class'] = [classes_names[int(f)] for f in gt_labels]#[f.replace("+","\n") for f in classes_names]

    df_subset['tsne-2d-one'] = (df_subset['tsne-2d-one'] - np.min(df_subset['tsne-2d-one'])) / (np.max(df_subset['tsne-2d-one']) - np.min(df_subset['tsne-2d-one']))
    df_subset['tsne-2d-two'] = (df_subset['tsne-2d-two'] - np.min(df_subset['tsne-2d-two'])) / (np.max(df_subset['tsne-2d-two']) - np.min(df_subset['tsne-2d-two']))

    #df_subset['tsne-2d-one']  = np.interp(df_subset['tsne-2d-one'] , (0, 1), (-180, 180))
    #df_subset['tsne-2d-two']  = np.interp(df_subset['tsne-2d-two'] , (0, 1), (-180, 180))

    #df_subset['tsne-2d-one'] = np.log(df_subset['tsne-2d-one'] + 1)  # Aggiungi 1 per evitare log(0)
    #df_subset['tsne-2d-two'] = np.log(df_subset['tsne-2d-two'] + 1)  # Aggiungi 1 per evitare log(0)

    plt.figure(figsize=(14, 10))
    sns.set_context('paper', font_scale=1.5)
    ax=sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=sns.color_palette("hls", len(list(np.unique(gt_labels)))),
        data=df_subset,
        legend=False,
        style="class",
        alpha=1, s=300
    )

    #plt.title('Awa2 Attribute Prototypes')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    size=50



    def label_point(x, y, val):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        added=[]
        for i, point in a.iterrows():

            x= point['x']
            y= point['y']
            for j in added:
                if abs(j[0] - x)<=0.05 and  abs(j[1]-y)<=0.05 :

                    x-=0.05

                    y-= 0.05


            added.append((x, y))

            plt.text(x, y, str(point['val']),horizontalalignment='center')

    #label_point(pd.Series(list(df_subset['tsne-2d-one'])),pd.Series(list(df_subset['tsne-2d-two'])), pd.Series(df_subset['class']))

    # plt.figure(dpi=400)

    plt.savefig(join("..","features",weight_path.split("/")[-1].split(".")[0]+".png"))
    plt.savefig(join("..","features",weight_path.split("/")[-1].split(".")[0]+".pdf"), bbox_inches='tight')
    plt.show()
    print("saved")

    print("dfdfd")