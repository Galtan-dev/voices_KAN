import opensmile
import toolbar_kan as tbk
from sklearn.decomposition import PCA

path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\healthy\\svdadult0002_healthy_50000.txt"
x = tbk.data_from_txt(path)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)
y = smile.process_signal(x, 50000)

print(y)
print(y.info())
print(y.describe().T)






# # mutual information
# from sklearn.feature_selection import mutual_info_classif
# mi = mutual_info_classif(X, y)

# PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# principal_components = pca.fit_transform(X)

# # L1 regularization
# from sklearn.linear_model import Lasso
# model = Lasso(alpha=0.01)
# model.fit(X, y)
# selected_features = model.coef_

# náhodný les
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X, y)
# importance = model.feature_importances_

# rfe
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# rfe = RFE(model, n_features_to_select=10)
# selected_features = rfe.fit_transform(X, y)

# select best
# from sklearn.feature_selection import SelectKBest, f_classif
# selector = SelectKBest(score_func=f_classif, k=10)
# selected_features = selector.fit_transform(X, y)

# variance treshold
# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(threshold=0.01)
# selected_features = selector.fit_transform(X)
