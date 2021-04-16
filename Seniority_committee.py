import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score


class Seniority_committee:
    """
        ML Algo based on the seniority committee method
    """
    
    def __init__(self, N):
        """
            :param N: число наблюдений, находящихся выше гиперплоскости
        """
        
        self.X_train = None
        self.y_train = None
        self.L = -1
        self.N = N
        self.weights_hp = dict() # здесь будут записаны оптимальные веса гиперплоскостей в виде: 
                                 # {'num_hyperplane': (voted_class, optim_weights)}
        self.optim_params = dict() # здесь будут записаны параметры оптимизации гиперплоскостей в виде:
                   # {'num_hyperplane': {'cycle_range': int, 'disp': bool, 'adaptive': bool, 'maxiter': int or None, 'xatol': int or None}}
                   # значение параметров дано в описании метода make_hyperplane
        
        
    def probability(self, X, w):
        """
            Принимает на вход матрицу фичей и вектор весов
            Возвращает предсказание вероятность того, что y = 1 при фиксированных x, P(y=1|x)

            :param X: матрица признаков 
            :param w: вектор весов
            :returns: вероятность того, что y = 1 при фиксированных x, P(y=1|x) ## вектор вероятностей = ReLU1(X.T*w)
        """
        
        linear = np.dot(X, w)
        linear[linear < 0] = 0
        linear[linear > 1] = 1
        
        return linear
    
    
    def compute_loss(self, X, y, w):
        """
            Принимает на вход матрицу весов, вектор ответов, вектор весов и параметр L, 
            влияющий на долю класса 0 в отсекающей гиперплоскости.
            Выдаёт на выход значение функции потерь
            
            :param X: матрица признаков
            :param w: вектор целевой переменной
            :param w: вектор весов
            :returns: значение функции потерь
        """
        
        p1 = probability(X, w)
        loss = np.sum((self.L - (self.L + 1) * y) * p1)
        
        return loss
    
    
    def compute_train_loss_class_0(self, w):
        """
            Function that we want to minimize, the committee member votes for class 1
            
            :param w: вектор весов
            :returns: значение функции потерь на обучающей выборке, когда член коммитета голосует за класс 0
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Model is not fitted')
        
        return compute_loss(self.X_train, 1 - self.y_train, w)
    
    
    def compute_train_loss_class_1(self, w):
        """
            Function that we want to minimize, the committee member votes for class 1
            
            :param w: вектор весов
            :returns: значение функции потерь на обучающей выборке, когда член коммитета голосует за класс 1
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Model is not fitted')
        
        return compute_loss(self.X_train, self.y_train, w)
    
    
    def make_hyperplane(self, class_num, X_train, c=0.1, cycle_range=100, disp=False, \
                        adaptive=True, maxiter=None, xatol=None):
        """
            Function that makes one of three hyperplanes
            
            :param class_num: класс, за который голосует данный член комитета: [0, 1]
            :param X_train: матрица признаков обучающей выборки
            :param number_of_hyperplane: порядковый номер гиперплоскости: [1, 2, 3]
            :param с: вспомогательный коэффициент для выбора начального приближения
            :param cycle_range: количество итераций минимизации функции потерь
            Параметры оптимизации с помощью алгоритма Нелдера-Мида:
                :param disp: bool: печать сообщения о сходимости
                :param adaptive: bool: адаптация параметров алгоритма для размерности задачи (полезно при больших размерностях)
                :param maxiter: максимально допустимое количество итераций при оптимизации
                :param xatol: абсолютная ошибка на оптимальных точках между итерациями, приемлемая для сходимости
            :returns: значение функции потерь на тестовой выборке
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Using make_hyperplane method before fitting')
        if class_num not in [0, 1]:
            raise Exception('Only binary classification is available, class_num should be 0 or 1')
        
        optim_result = []
        start_time = time.time()

        optim_result_new = []

        for i in range(cycle_range):
            start_w = np.array((np.random.rand(X_test.shape[1])-0.5)*c)
            if class_num == 1:
                res = minimize(compute_train_loss_class_1, x0=start_w, method='Nelder-Mead', \
                               options={'disp': disp,'adaptive': adaptive, 'maxiter': maxiter, 'xatol': xatol})
            elif class_num == 0:
                res = minimize(compute_train_loss_class_0, x0=start_w, method='Nelder-Mead', \
                               options={'disp': disp,'adaptive': adaptive, 'maxiter': maxiter, 'xatol': xatol})
            if res.fun < 0: # ??? - вообще не универсально      
                optim_result.append(res.x)

        # Имеет ли большой смысл следующее???
        for start_w_new in optim_result:
            if class_num == 1:
                res = minimize(compute_train_loss_class_1, x0=start_w_new, method='Nelder-Mead', \
                         options={'disp': False, 'adaptive':True, 'xatol':1})
            elif class_num == 0:
                res = minimize(compute_train_loss_class_0, x0=start_w_new, method='Nelder-Mead', \
                         options={'disp': False, 'adaptive':True, 'xatol':1})

            for k in range(2):
                if class_num == 1:
                    res = minimize(compute_train_loss_class_1, x0=res.x, method='Nelder-Mead', \
                             options={'disp': False, 'adaptive':True})
                elif class_num == 0:
                    res = minimize(compute_train_loss_class_0, x0=res.x, method='Nelder-Mead', \
                             options={'disp': False, 'adaptive':True})

            optim_result_new += [[res.fun, res.x]]

        optim_result_new = pd.DataFrame(optim_result_new)
        optim_result = optim_result.sort_values(0).head(1)
        hyperplane_coefficients = optim_result[1].values[0]
        print('Time taken for optimization: {0}'.format(time.time() - start_time))
#                 print('The best result was on the step {0}'.format(optim_result[0].values[0]))
        print('The minimum of the loss function: {0}'.format(optim_result[0].values[0]))

#             self.weights_hp_3 = hyperplane_coefficients
        return hyperplane_coefficients
        
    
    def cutter(self, X, w):
        """
            Function that makes binary targets for rational numbers
            
            :param X: матрица признаков
            :param w: вектор оптимальных весов
            :returns: бинаризованные предсказания целевой переменной
        """
        linear = np.dot(X, w)
        linear[linear < 0] = 0
        return np.sign(linear)
    
    
    def fit(self, X, y):
        """
            Fits the algorithm on the train sample 
            
            :param X: матрица признаков, обучающая выборка
            :param y: вектор истинных значений целевой переменной обучающей выборки
            :returns: -, но на выходе обученная моделька
        """
        
        self.X_train = X
        self.y_train = y
        self.optim_params = {'cycle_range': 1000, 'disp': False, 'adaptive': True, 'maxiter': 100, 'xatol': 0.3}
#         preds = np.zeros(len(y))
#         train_preds = pd.DataFrame(y, columns=['TARGET'])
#         train_preds['PEDICTIONS'] = 0
        
        for hp_num in range(1, 4):
            
            k = 10
            
            while True:
                
                self.L = 2 ** k
                
                hp_weights_class_1 = make_hyperplane(class_num=1, self.X_train, \
                                                     c=0.1, cycle_range=optim_params['cycle_range'], \
                                                     disp=optim_params['disp'], \
                                                     adaptive=optim_params['adaptive'], \
                                                     maxiter=optim_params['maxiter'], \
                                                     xatol=optim_params['xatol'])
                
                hp_weights_class_0 = make_hyperplane(class_num=0, self.X_train, \
                                                     c=0.1, cycle_range=optim_params['cycle_range'], \
                                                     disp=optim_params['disp'], \
                                                     adaptive=optim_params['adaptive'], \
                                                     maxiter=optim_params['maxiter'], \
                                                     xatol=optim_params['xatol'])
                
                cut_1 = cutter(self.X_train, hp_weights_class_1)
                cut_0 = cutter(self.X_train, hp_weights_class_0)
                X_1 = self.X_train[cut_1 == 1]
                X_0 = self.X_train[cut_0 == 0]
                
                if X_1.shape[0] < self.N and X_0.shape[0] < self.N:
                    k -= 1
                    continue
                else:
                    if X_1.shape[0] > X_0.shape[0]:
                        self.weights_hp[hp_num] = (1, hp_weights_class_1)
#                         X_class_1 = self.X_train[cut_1 == 1]
#                         preds[cut_1 == 1] = probability(X_class_1, hp_weights_class_1)
                        
                        self.X_train = self.X_train[cut_1 == 0]
                        self.y_train = self.y_train[cut_1 == 0]
                        
                    else X_1.shape[0] > X_0.shape[0]:
                        self.weights_hp[hp_num] = (0, hp_weights_class_0)
#                         X_class_0 = self.X_train[cut_0 == 0]
#                         preds[cut_0 == 0] = probability(X_class_0, hp_weights_class_0)
                        
                        self.X_train = self.X_train[cut_0 == 1]
                        self.y_train = self.y_train[cut_0 == 1]
                    break
                
    # TODO:
    def predict(self, X):
        """
            Makes predict_proba for the sample
            
            :param X: матрица признаков, обучающая выборка
            :returns: предсказания вероятностей отнесения к классу 1
        """
        test_ = X.copy()
        if self.weights_hp == {}:
            raise Exception('Model is not fitted')
        
        predictions = np.zeros(X.shape[0])
        
        for hp_num in range(1, 4):
            
            class_num, weights = self.weights_hp[hp_num]
            
            cut = cutter(self.X_train, hp_weights_class_1)
            
            if class_num == 1:
                X_ = self.X_train[cut_1 == 1]
            elif class_num == 0:
                X_ = self.X_train[cut_0 == 0]
            
            prob = probability(X_, weights)
            
            j = 0
            k = 0
            for i in range(len(predictions)):
                
                if predictions[i] == 0:
                    if cut[k] == True:
                        predictions[i] = prob[j]
                        j += 1
                    k += 1
            return predictions
            
                                                 
        