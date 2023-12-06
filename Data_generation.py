import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_object_dtype
import patsy
from scipy.linalg import block_diag

np.random.seed(1)

def create_dataset(N=200, p=3, M= None, mu_x = None, sigma_x = None):
    # p_r:          number of (continuous) independent variables
    # sigma_x:      variances of indpendent variables (uncorrelated!)
    # N:            number of total observations
    # M:            number of crossed random effects per grouping factor
    b = len(M)              #number of grouping columns

    cols =[]
    for i in range(p):
        cols = np.append(cols, "x"+ str(i))
    for j in range(b):
        cols = np.append(cols, "group"+ str(j))
    data = pd.DataFrame(np.empty((N,p+b)),columns=cols)

    #if mu and sigma are not specified for X, use standard gaussian distr
    # the columns to put into X should be uncorrelated (otherwise: multicollinearity might yield some problems)
    if mu_x is None:
        mu_x = np.zeros(p)
    if sigma_x is None:
        sigma_x = np.eye(p)
    X = np.random.multivariate_normal(mu_x, sigma_x, N)
    data.iloc[:,:p] = X

    #here: balanced design, i.e. we have about the same number of observations in all category-combinations (nij identical)
    i = 0
    for j in M:
        if i ==0:
            g_q = np.random.randint(low=1, high=j + 1, size=N)
            g = g_q
        else:
            g_q = np.random.randint(low=1, high=j + 1, size=N)
            g= np.stack((g, g_q), axis=1)
        i+=1
    g = g.astype(int)
    data.iloc[:,p:] = g.astype(str)
    return data


def build_Z(data, formula_Z):

    #create Z:
    dmat = patsy.dmatrix(formula_Z, data)

    di = dmat.design_info
    ods = di.term_codings


    j = 0
    list_levels = []
    l_group=[]
    for key in ods.items():
        try:
            z = list(ods)[j].factors[0]
            if not [key[1][0].contrast_matrices.get(z).column_suffixes][0][0].endswith('1]'):
                cat_suff = [key[1][0].contrast_matrices.get(z).column_suffixes][0][0].split("]")
                [key[1][0].contrast_matrices.get(z).column_suffixes][0].insert(0, cat_suff[0][:-1] + '1]')
            l_group = l_group + [z]
            list_levels = list_levels + [key[1][0].contrast_matrices.get(z).column_suffixes]
        except IndexError:
            print()
        j += 1

    n_groups_Z =  len(dict.fromkeys(l_group))

    term_names = di.term_names
    if term_names[0] == 'Intercept':
        term_names = term_names[1:]

    list_cols = []
    list_cat = []

    j = 0

    for term_name in term_names:
        if not ":" in term_name:
            list_cat_j = []
            for cat in list_levels[j]:
                list_cat_j = list_cat_j + [cat]
                list_cols = list_cols + [term_name + cat]
                if any(term_name + ":" in s for s in term_names):
                    matching = [s for s in term_names if any(xs in s for xs in [term_name + ":"])]
                    for m in matching:
                        splitted = m.split(':')
                        if splitted[0] + cat + ':' + splitted[1] in di.column_names and splitted[0] + cat + ':' + \
                                splitted[1] not in list_cols:
                            list_cols = list_cols + [splitted[0] + cat + ':' + splitted[1]]
                        else:
                            list_cols = list_cols + [splitted[0] + cat[:1] + cat[3:] + ':' + splitted[1]]
            list_cat = list_cat + [list_cat_j]

        else:  # no random intercept. only random slope
            list_cat_j = []
            flag=0
            for cat in list_levels[j]:
                splitted = term_name.split(':')
                matching = [s for s in term_names if any(xs in s for xs in [splitted[0] + ':'])]
                if splitted[0] + cat + ':' + splitted[1] in di.column_names and splitted[0] + cat + ':' + splitted[
                    1] not in list_cols:
                    for m in matching:
                        splitted = m.split(':')
                        list_cols = list_cols + [splitted[0] + cat + ':' + splitted[1]]
                    list_cat_j = list_cat_j + [cat]
                    flag=1
            if flag==1:
                list_cat = list_cat + [list_cat_j]
        j += 1



    if len(list_cols) != len(di.column_names) and di.column_names[0] != 'Intercept':
        print("lenERROR!")
    colnames_orig = di.column_names
    dmat_df = pd.DataFrame(dmat, columns=colnames_orig)

    for z in list_cols:
        if z not in colnames_orig:
            dmat_df[z] = 0

            if dmat_df.columns[0] == 'Intercept':
                dmat_df = dmat_df.drop(dmat_df.columns[0], axis=1)

            cols = dmat_df.columns
            sub_df = dmat_df.loc[:,[categ.startswith(z.split('[')[0]) and ':' not in categ and categ != z for categ in cols]]
            dmat_df.loc[(dmat_df.loc[:, sub_df.columns] != 1).all(axis=1), z] = 1

    Z = dmat_df[list_cols]
    return Z, n_groups_Z, list_cat


def make_design_matrices(data, cols_X=None, Z=None, formula_Z =None, with_interc_X = True):
    # input for cols_X can either be a list of strings, or a list of integers
    # if we do not specify the column names to use for X and Z, by default choose all columns created for X and Z
    N = data.shape[0]
    if cols_X is None:
        cols_X = [col for col in data if col.startswith('x')]
    cols_X_cont = [col for col in cols_X if is_float_dtype(data[col])]
    cols_X_nom = [col for col in cols_X if is_object_dtype(data[col])]

    # make design matrix X
    if with_interc_X is True:
        X = np.ones((N, 1))
        X = np.hstack((X, data[cols_X_cont]))
    else:
        X = data[cols_X_cont]
    if len(cols_X_nom) >=1:
        # make dummy matrix
        X_dummy = pd.get_dummies(data=data[cols_X_nom], drop_first=False)
        X = np.matrix(pd.concat([pd.DataFrame(X), X_dummy], axis=1))

    if Z is None:
        Z, n_groups_Z, list_cat = build_Z(data=data, formula_Z=formula_Z)

    return np.asarray(X), np.asarray(Z), n_groups_Z, list_cat





def create_G(sigma_sq = None, Psi=None, Z=None, categories = None, formula_Z=None, data=None):
    #Input:
    #sigma_square: scale factor
    #Psi: list of within-group covariance matrices of sizes [q0 x q0,.... q_b x q_b]. (e.g. 2x2 for random intercept and 1 random slope)
    # either Z and categories or formula_Z and data must be specified:
        #Z: Z already ordered in design gr0[1] Int | gr0[1] Slope .... gr0[M_0] Int | gr0[M_0] Slopes gr1[1] Intercept | gr1[1] Slopes ... gr1[M_1] Intercept | gr1[M_1] Slopes
            #if not available in this form: first build by function above by using formula_Z
            #categories: M=[M_0, M_1, ..., M_b]
        #formula_Z: a string of type 'formula' with variables as columns in data.

    if Z is None:
        Z, n_groups_Z, categories =  build_Z(data=data,formula_Z=formula_Z)

    j=0
    for cat in categories:
        Psi_gr = block_diag(*([Psi[j]]*len(cat)))
        if j==0:
            G = Psi_gr
        else:
            G = block_diag(G, Psi_gr)
        j+=1
    if G.shape[1] != Z.shape[1]:
        print("DimensionError: Dimensions of Psi don't match dimensions of design matrix Z")
    return sigma_sq, G


def create_responses(X,Z, G, sigma_sq, beta, data):

    #create realizations of u
    n = X.shape[0]
    p = X.shape[1]
    q = G.shape[1]

    u = np.random.multivariate_normal(np.zeros((q)), G, 1).T
    epsilon = np.random.normal(0, sigma_sq, size=n)
    Xbeta = np.matmul(X, beta)
    Zu = np.squeeze(np.matmul(Z, u), axis=1)
    # make data y = Xbeta+Zu + eps
    y = Xbeta +  Zu + epsilon
    data['y'] = y
    return data, np.expand_dims(np.asarray(data['y']), axis=1)



M=(3,6)
dataset = create_dataset(p=2,M=M)
cols_X = ['x0', 'x1']


formula =  "group0/x0 + group1"


X, Z, n_groups_Z, list_cat = make_design_matrices(dataset, cols_X=cols_X, formula_Z=formula)
Psi = [np.array([[1,0.5], [0.5,1]]), np.array([3])]
sigma_sq = 0.1
sigma_sq, G = create_G(sigma_sq = sigma_sq, Psi=Psi,Z=Z, categories = list_cat, formula_Z=formula)


beta = np.array([50.0, 15.0, 20.0])
full_dataset, y = create_responses(X=X, Z=Z,G=G, sigma_sq=sigma_sq, beta=beta, data=dataset)
theta0 = np.exp(0.1), G

