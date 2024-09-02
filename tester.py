# Standard libraries
import numpy as np  
import pandas as pd  
from collections import Counter  

# Machine Learning libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from catboost import CatBoostClassifier

# Custom imports from your project structure
from models.E_SMOTE_ADASVM_TW.model import E_SMOTE_ADASVM_TW  
from models.ADASVM_TW.model import ADASVM_TW 
from models.KELM.model import KELM  
from models.KELM.param_opt import *
from auxiliary.test import  test
from auxiliary.plotter import * 
from auxiliary.CUS import * 




def column_average(df, column_name):
    # Stack the arrays in the specified column to form a 2D array
    stacked_array = np.stack(df[column_name].values)
    
    # Calculate the mean along the rows (axis=0)
    column_means = np.mean(stacked_array, axis=0)
    
    return column_means
    


def test_models(data1, var1, n_test = 3, test_year = 2016, percentage = 0.1, k = 100, isTest = False, balanced = False):
    test_res = {
        'E-SMOTE-ADASVM-TW': [],
        'CATBOOST': [],
        'LSEOFOA-KELM': [],
        'GWO-KELM': [],
        'GP': [],
        'GDBT': [],
        'RF': [],
        'MLP': [],
        'SVM': [],
        'LR': []
        
    }
    cs = []
    gammas = []
    random_states = np.random.randint(1, 9999, size = n_test)
    X_test = data1[(data1['fyear'] > test_year) & (data1['fyear'] <= 2021)][var1].values
    y_test = data1[(data1['fyear'] > test_year) & (data1['fyear'] <= 2021)]['h3'].values
    if isTest:
        n_test = 1
    for i in range(n_test):
        if isTest:
            X_opt, y_opt, t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                                  year = test_year,minority_percentage = 0.1, percentage=0.01, minority = True, random_state= random_states[i])
            X1,y1,t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                          year = test_year,minority_percentage = 0.1, percentage=0.01, minority = True, random_state= random_states[i])
        else:    
            X_opt, y_opt, t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                                      year = test_year,minority_percentage = percentage,percentage=0.01, minority = True, random_state= random_states[i])
            if not balanced:
                
                X1,y1,t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                              year = test_year,percentage=percentage, random_state= random_states[i])
            else:
                count = Counter(data1['h3'].values)
                rat = count[1] / count[-1]
                X1,y1,t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                              year = test_year,percentage=rat, random_state= random_states[i])
        
        ## E_SMOTE_ADASVM_TW
        #print("E_SMOTE_ADASVM_TW...")
        
        e_model = E_SMOTE_ADASVM_TW(t)
        e_model.fit(X1,y1)
        
        test_res['E-SMOTE-ADASVM-TW'].append(test(e_model, X_test, y_test, verbose=False))

        
        ## Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X1,y1)
        test_res['RF'].append(test(rf_model, X_test, y_test, verbose=False))

        
        ## MLP
        mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32),  # Two hidden layers with 64 and 32 neurons
                      activation='relu',            # Activation function
                      solver='adam',                # Optimization algorithm
                      max_iter=500)
        mlp_model.fit(X1,y1)
        test_res['MLP'].append(test(mlp_model, X_test, y_test, verbose=False))

        
        ## SVM
        svm_model = SVC(probability= True)
        svm_model.fit(X1,y1)
        test_res['SVM'].append(test(svm_model, X_test, y_test, verbose=False))

        
        ## LogReg
        lr_model = LogisticRegression()
        lr_model.fit(X1,y1)
        test_res['LR'].append(test(lr_model, X_test, y_test, verbose=False))

        
        ## CATBOOST
        #print("CATBOOST...")
        
        ca_model = CatBoostClassifier(iterations=100,
                                   depth=8,
                                   learning_rate=0.1,
                                   l2_leaf_reg = 0.1,
                                   rsm = 0.95,
                                   border_count = 64,
                                   verbose = False)
        ca_model.fit(X1,y1)
        test_res['CATBOOST'].append(test(ca_model,X_test,y_test, verbose=False))
        
        ## LSEOFOA-KELM
        #print("LSEOFOA_KELM...")
        
        c,gamma = LSEOFOA(X_opt, y_opt , R = 0.8, z = 0.6, bounds = (2**(-5), 2**5), max_iters = 32, size_pop = 5)
        gammas.append(gamma)
        lsk_model = KELM(C=c, gamma=gamma)
        lsk_model.fit(X1,y1)
        test_res['LSEOFOA-KELM'].append(test(lsk_model,X_test,y_test, verbose=False))
        
        ## GWO-KELM
        #print("GWO_KELM...")
        
        c1,gamma1 = GWO(X_opt, y_opt, bounds = (2**(-5), 2**5), max_iters = 32, size_pop = 10)
        gammas.append(gamma1)
        gwk_model = KELM(C=c1, gamma=gamma1)
        gwk_model.fit(X1,y1)
        test_res['GWO-KELM'].append(test(gwk_model,X_test,y_test, verbose=False))
        
        ## GP
        #print("GP...")
        kernel = 1.0 * RBF(1.0)
        gp = GaussianProcessClassifier(kernel=kernel)
        gp.fit(X1,y1)
        test_res['GP'].append(test(gp,X_test, y_test, verbose=False))
        
        ## CUSBOOST
        #print("CUSBOOST...")
        cus_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
        cus_model.fit(X1,y1)
        test_res['GDBT'].append(test(cus_model, X_test, y_test, verbose=False))

        
    test_res = pd.DataFrame(test_res)    
    test_res =test_res.applymap(lambda x: np.array(x))
    plot_boxes(test_res)
    averages = {col: column_average(test_res, col) for col in test_res.columns}
    res = pd.DataFrame(np.stack(list(averages.values())),columns=['ACC', 'Recall', 'Precision', 'F1','MCC','AUC'])
    res['Model'] = list(averages.keys())
    res = res[['Model'] + ['ACC', 'Recall', 'Precision', 'F1','MCC','AUC']]
    return res

def test_models_total(datas, variables, names, n_test = 3, test_year = 2016, percentage = 0.1, k = 100, isTest = False, balanced = False):
    results = []
    total_res = {
            'E-SMOTE-ADASVM-TW': [],
            'CATBOOST': [],
            'LSEOFOA-KELM': [],
            'GWO-KELM': [],
            'GP': [],
            'GDBT': [],
            'RF': [],
            'MLP': [],
            'SVM': [],
            'LR': []         
        }
    for data1, var1, name in zip(datas,variables, names):
    
        test_res = {
            'E-SMOTE-ADASVM-TW': [],
            'CATBOOST': [],
            'LSEOFOA-KELM': [],
            'GWO-KELM': [],
            'GP': [],
            'GDBT': [],
            'RF': [],
            'MLP': [],
            'SVM': [],
            'LR': []
            
        }
        rocs = {
            'E-SMOTE-ADASVM-TW': [],
            'CATBOOST': [],
            'LSEOFOA-KELM': [],
            'GWO-KELM': [],
            'GP': [],
            'GDBT': []#,
            #'RF': [],
            #'MLP': [],
            #'SVM': [],
            #'LR': []         
        }
        mean_fpr = np.linspace(0, 1, 100)
        cs = []
        gammas = []
        random_states = np.random.randint(1, 9999, size = n_test)
        X_test = data1[(data1['fyear'] > test_year) & (data1['fyear'] <= 2021)][var1].values
        y_test = data1[(data1['fyear'] > test_year) & (data1['fyear'] <= 2021)]['h3'].values
        if isTest:
            n_test = 1
        for i in range(n_test):
            if isTest:
                X_opt, y_opt, t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                                      year = test_year,minority_percentage = 0.1, percentage=0.01, minority = True, random_state= random_states[i])
                X1,y1,t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                              year = test_year,minority_percentage = 0.1, percentage=0.01, minority = True, random_state= random_states[i])
            else:    
                X_opt, y_opt, t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                                      year = test_year,minority_percentage = percentage,percentage=0.01, minority = True, random_state= random_states[i])
                if not balanced:
                    
                    X1,y1,t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                                  year = test_year,percentage=percentage, random_state= random_states[i])
                else:
                    count = Counter(data1['h3'].values)
                    rat = count[1] / count[-1]
                    X1,y1,t = CUS(data1[data1['fyear'] <= test_year], var1, 'h3', k=k, tw = True,
                                  year = test_year,percentage=rat, random_state= random_states[i])
            
            ## E_SMOTE_ADASVM_TW
            #print("E_SMOTE_ADASVM_TW...")
            if not balanced:
                e_model = E_SMOTE_ADASVM_TW(t)
            else:
                e_model = ADASVM_TW(t)
                
            e_model.fit(X1,y1)
            tesm,rocesm = test(e_model, X_test, y_test, verbose=False)
            test_res['E-SMOTE-ADASVM-TW'].append(tesm)
            total_res['E-SMOTE-ADASVM-TW'].append(tesm)
            rocesm = np.interp(mean_fpr, rocesm[0], rocesm[1])
            rocesm[0] = 0.0
            rocs['E-SMOTE-ADASVM-TW'].append(rocesm)
            
            ## Random Forest
            rf_model = RandomForestClassifier()
            rf_model.fit(X1,y1)
            trf, rocrf = test(rf_model, X_test, y_test, verbose=False)
            test_res['RF'].append(trf)
            total_res['RF'].append(trf)
            #rocrf = np.interp(mean_fpr, rocrf[0], rocrf[1])
            #rocrf[0] = 0.0
            #rocs['RF'].append(rocrf)
            
            ## MLP
            mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32),  # Two hidden layers with 64 and 32 neurons
                          activation='relu',            # Activation function
                          solver='adam',                # Optimization algorithm
                          max_iter=500)
            mlp_model.fit(X1,y1)
            tmlp,rocmlp = test(mlp_model, X_test, y_test, verbose=False)
            test_res['MLP'].append(tmlp)
            total_res['MLP'].append(tmlp)
            #rocmlp = np.interp(mean_fpr, rocmlp[0], rocmlp[1])
            #rocmlp[0] = 0.0
            #rocs['MLP'].append(rocmlp)
            
            ## SVM
            svm_model = SVC(probability= True)
            svm_model.fit(X1,y1)
            tsvm,rocsvm = test(svm_model, X_test, y_test, verbose=False)
            test_res['SVM'].append(tsvm)
            total_res['SVM'].append(tsvm)
            #rocsvm = np.interp(mean_fpr, rocsvm[0], rocsvm[1])
            #rocsvm[0] = 0.0
            #rocs['SVM'].append(rocsvm)
            
            ## LogReg
            lr_model = LogisticRegression()
            lr_model.fit(X1,y1)
            tlr, roclr = test(lr_model, X_test, y_test, verbose=False)
            test_res['LR'].append(tlr)
            total_res['LR'].append(tlr)
            #roclr = np.interp(mean_fpr, roclr[0], roclr[1])
            #roclr[0] = 0.0
            #rocs['LR'].append(roclr)
            
            ## CATBOOST
            #print("CATBOOST...")
            
            ca_model = CatBoostClassifier(iterations=100,
                                       depth=8,
                                       learning_rate=0.1,
                                       l2_leaf_reg = 0.1,
                                       rsm = 0.95,
                                       border_count = 64,
                                       verbose = False)
            ca_model.fit(X1,y1)
            tcat, roccat = test(ca_model,X_test,y_test, verbose=False)
            test_res['CATBOOST'].append(tcat)
            total_res['CATBOOST'].append(tcat)
            roccat = np.interp(mean_fpr, roccat[0], roccat[1])
            roccat[0] = 0.0
            rocs['CATBOOST'].append(roccat)
            
            ## LSEOFOA-KELM
            #print("LSEOFOA_KELM...")
            
            c,gamma = LSEOFOA(X_opt, y_opt , R = 0.8, z = 0.6, bounds = (2**(-5), 2**5), max_iters = 32, size_pop = 5)
            gammas.append(gamma)
            lsk_model = KELM(C=c, gamma=gamma)
            lsk_model.fit(X1,y1)
            tlsk, roclsk = test(lsk_model,X_test,y_test, verbose=False)
            test_res['LSEOFOA-KELM'].append(tlsk)
            total_res['LSEOFOA-KELM'].append(tlsk)
            roclsk = np.interp(mean_fpr, roclsk[0], roclsk[1])
            roclsk[0] = 0.0
            rocs['LSEOFOA-KELM'].append(roclsk)
            ## GWO-KELM
            #print("GWO_KELM...")
            
            c1,gamma1 = GWO(X_opt, y_opt, bounds = (2**(-5), 2**5), max_iters = 32, size_pop = 10)
            gammas.append(gamma1)
            gwk_model = KELM(C=c1, gamma=gamma1)
            gwk_model.fit(X1,y1)
            tgwk, rocgwk = test(gwk_model,X_test,y_test, verbose=False)
            test_res['GWO-KELM'].append(tgwk)
            total_res['GWO-KELM'].append(tgwk)
            rocgwk = np.interp(mean_fpr, rocgwk[0], rocgwk[1])
            rocgwk[0] = 0.0
            rocs['GWO-KELM'].append(rocgwk)
            
            ## GP
            #print("GP...")
            kernel = 1.0 * RBF(1.0)
            gp = GaussianProcessClassifier(kernel=kernel)
            gp.fit(X1,y1)
            tgp, rocgp = test(gp,X_test, y_test, verbose=False)
            test_res['GP'].append(tgp)
            total_res['GP'].append(tgp)
            rocgp = np.interp(mean_fpr, rocgp[0], rocgp[1])
            rocgp[0] = 0.0
            rocs['GP'].append(rocgp)
            
            ## CUSBOOST
            #print("CUSBOOST...")
            cus_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
            cus_model.fit(X1,y1)
            tcus,roccus = test(cus_model, X_test, y_test, verbose=False)
            test_res['GDBT'].append(tcus)
            total_res['GDBT'].append(tcus)
            roccus = np.interp(mean_fpr, roccus[0], roccus[1])
            roccus[0] = 0.0
            rocs['GDBT'].append(roccus)
            
        test_res = pd.DataFrame(test_res)    
        test_res =test_res.applymap(lambda x: np.array(x))
        
        
        plot_rocs(rocs, name)
        
        averages = {col: column_average(test_res, col) for col in test_res.columns}
        res = pd.DataFrame(np.stack(list(averages.values())),columns=['ACC', 'Recall', 'Precision', 'F1','MCC','AUC'])
        res['Model'] = list(averages.keys())
        res = res[['Model'] + ['ACC', 'Recall', 'Precision', 'F1','MCC','AUC']]
        results.append(res)
    total_res = pd.DataFrame(total_res)    
    total_res =total_res.applymap(lambda x: np.array(x))
    plot_boxes(total_res)
    return results