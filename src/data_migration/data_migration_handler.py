import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


def encode_and_bind(original_dataframe, feature_to_encode, ans_modalities,m):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]],dummy_na=True)
    res = pd.concat([original_dataframe, dummies],axis=1)
    res = res.drop([feature_to_encode], axis=1)
    ans_modalities[m].remove(feature_to_encode)
    ans_modalities[m]+=list(dummies.columns)

    return res, ans_modalities


def extract_text_features(X_unlabled):
    output = X_unlabled.copy()
    output.columns = [c.split(': ')[1] if ': ' in c else c for c in output.columns]
    a = pd.melt(output.reset_index(), id_vars=['drugBank_id'])
    a.value = a.value.astype(bool)
    a = a[a.value == True]
    a = a.groupby('drugBank_id').variable.apply(lambda x: "%s " % ' '.join(x))

    count_vect = CountVectorizer(binary=True)
    a = a.fillna('')
    word_counts = count_vect.fit_transform(a)
    text_features = ['Mention: ' + x for x in count_vect.get_feature_names()]
    text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=a.index)
    a = pd.DataFrame(index=X_unlabled.index).join(text_features, how='left').fillna('')
    return a


def add_mods(X,X2,mods,modality_name):
    for c in X2.columns:
        mods = mods.append({'modality': modality_name, 'feature': c}, ignore_index=True)
    X = X.join(X2, how='left')
    return X, mods


def get_col_clusters(X_unlabled,number_of_clusters):
    s = pd.Series(X_unlabled.columns, index=X_unlabled.columns)
    s = s[~s.str.contains('Number of')]
    newvals = [c.split(': ')[1] if ': ' in c else c for c in s.values]
    s = pd.Series(newvals, index=s.index)
    count_vect = CountVectorizer(binary=True)
    word_counts = count_vect.fit_transform(s)
    word_counts = word_counts
    text_features = count_vect.get_feature_names()
    text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=s.index)
    ans = None
    print('features,words:',text_features.values.shape)
    print(list(text_features.columns)[:1000])
    dist_mat = pairwise_distances(text_features.values, metric='jaccard', n_jobs=1) #jaccard
    print('done sim matrix')
    kmeans = AgglomerativeClustering(n_clusters=number_of_clusters,linkage='average',affinity='precomputed')#affinity=sklearn.metrics.jaccard_score ,affinity='cosine'
    clusters = pd.DataFrame(kmeans.fit_predict(dist_mat),columns=['cluster'],index=text_features.index)
    for g in clusters.groupby('cluster').groups:
        g= clusters.groupby('cluster').groups[g]
        col_name='Cluster '+str(number_of_clusters)+ ' :  ' + '; '.join([str(gr) for gr in g])
        g_col = X_unlabled[g].sum(axis=1).astype(bool)
        if ans is None:
            g_col.name = col_name
            ans = pd.DataFrame(g_col)
        else:
            ans[col_name] = g_col
    print('done clustering',number_of_clusters)

    print('done clustering')
    return ans