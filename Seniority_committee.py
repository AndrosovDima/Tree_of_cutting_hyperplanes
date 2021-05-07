import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import roc_auc_score

# ХОТЕЛКИ:

# ХОЧЕТСЯ ДОБАВИТЬ МЕТОД ДЛЯ СОХРАНЕНИЯ ВЕСОВ В ФАЙЛ И ЗАГРУЗКИ ИХ ИЗ ФАЙЛА
# ХОЧЕТСЯ ПРИ ОПТИМИЗАЦИИ СОРТИРОВАТЬ ТОЧКИ ПО ЗНАЧЕНИЮ ФУНКЦИЙ В ПОРЯДКЕ ВОЗРАСТАНИЯ И БРАТЬ ПЕРВЫЕ N ШТУК + помогло значительно уменьшить время обучения
# ХОЧЕТСЯ ПОИГРАТЬСЯ С ПАРАМЕТРАМИ ОПТИМИЗАЦИИ В ЦЕЛОМ +- пока что особо ничего
# ХОЧЕТСЯ ПОПРОБОВАТЬ ИСПОЛЬЗОВАТЬ ДРУГИЕ МЕТОДЫ ОПТИМИЗАЦИИ, НАПРИМЕР, ДИФФЕРЕНЦИАЛЬНУЮ ЭВОЛЮЦИЮ + помогло добиться лучших результатов, в частности метод TNC показал себя очень хорошо
# ХОЧЕТСЯ СРАВНИТЬ С ДРУГИМИ МОДЕЛЯМИ, НАПРИМЕР ЛОГРЕГ И ТД
# ХОЧЕТСЯ РАССМОТРЕТЬ РАБОТУ НА НЕСБАЛАНСИРОВАННОЙ ВЫБОРКЕ, ВОЗМОЖНО ПЛОХОЕ КАЧЕСТВО !!!!!!!!!!!!!!!!!!

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
                                 # {'num_hyperplane': (voted_class, optim_weights, probability)}
        self.optim_params = dict() # здесь будут записаны параметры оптимизации гиперплоскостей в виде:
                   # {'cycle_range': int, 'disp': bool, 'adaptive': bool, 'maxiter': int or None, 'xatol': float or None}
                   # значение параметров дано в описании метода make_hyperplane
        
    def expand(self, X):
        """
            Concatenates the feature matrix with the column of ones
            
            :param X: матрица признаков
            :returns: исходная матрица признаков со столбцом единиц (добавили bias)
        """
        X_new = X.copy()
        ones_col = np.ones(len(X_new))
        ones_col.shape = (len(X_new), 1)
        X_new = np.hstack((X_new, ones_col))
        return X_new
    
    
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
        
        p1 = self.probability(X, w)
        loss = np.sum((self.L - (self.L + 1) * y) * p1)
        
        return loss
    
    
    def compute_train_loss_class_0(self, w):
        """
            Function that we want to minimize, the committee member votes for class 0
            
            :param w: вектор весов
            :returns: значение функции потерь на обучающей выборке, когда член коммитета голосует за класс 0
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Model is not fitted')
        
        return self.compute_loss(self.X_train, 1 - self.y_train, w)
    
    
    def compute_train_loss_class_1(self, w):
        """
            Function that we want to minimize, the committee member votes for class 1
            
            :param w: вектор весов
            :returns: значение функции потерь на обучающей выборке, когда член коммитета голосует за класс 1
        """
        
        if self.X_train is None or self.y_train is None:
            raise Exception('Model is not fitted')
        
        return self.compute_loss(self.X_train, self.y_train, w)
    
    
    def make_hyperplane(self, class_num, X_train, optim_method, c=0.1, cycle_range=100, disp=False, \
                        adaptive=True, maxiter=None, xatol=None, verbose=0):
        """
            Function that makes one of three hyperplanes
            
            :param class_num: класс, за который голосует данный член комитета: [0, 1]
            :param X_train: матрица признаков обучающей выборки
            :param optim_method: метод оптимизации: ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP', \
                                                    'COBYLA', 'TNC']
            :param с: вспомогательный коэффициент для выбора начального приближения
            :param cycle_range: количество итераций минимизации функции потерь
            Параметры оптимизации с помощью алгоритма Нелдера-Мида:
                :param disp: bool: печать сообщения о сходимости
                :param adaptive: bool: адаптация параметров алгоритма для размерности задачи (полезно при больших размерностях)
                :param maxiter: максимально допустимое количество итераций при оптимизации
                :param xatol: абсолютная ошибка на оптимальных точках между итерациями, приемлемая для сходимости
            :param verbose: подробный вывод описания обучения: [0, 1, 2]:
                0 - не печатать ничего
                1 - печатать общее время обучения
                2 - подробный вывод информации о процессе обучения
            :returns: значение функции потерь на тестовой выборке
        """

        if self.X_train is None or self.y_train is None:
            raise Exception('Using make_hyperplane method before fitting')
        if class_num not in [0, 1]:
            raise Exception('Only binary classification is available, class_num should be 0 or 1')
        if optim_method not in ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP', \
                                                    'COBYLA', 'TNC']:
            raise Exception("Unavailable optimization method, only ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP', 'COBYLA', 'TNC'] are available")
        
        optim_result = []
        optim_result_more_precise = []
        start_time = time.time()
        
        if verbose == 2:
            print('Optimization is started')
        
        if optim_method == 'Nelder-Mead':
            
            for i in range(cycle_range):

                start_w = np.array((np.random.rand(X_train.shape[1]) - 0.5) * c)

                if class_num == 1:

                        res = minimize(self.compute_train_loss_class_1, x0=start_w, method='Nelder-Mead', \
                                       options={'disp': disp,'adaptive': adaptive, 'maxiter': maxiter, 'xatol': xatol})

                elif class_num == 0:

                        res = minimize(self.compute_train_loss_class_0, x0=start_w, method='Nelder-Mead', \
                                       options={'disp': disp,'adaptive': adaptive, 'maxiter': maxiter, 'xatol': xatol})

                optim_result.append([res.fun, res.x])


            if optim_result == []:
                raise Exception('We\'ve not get any satisfying first approximation')

            optim_result.sort(key=lambda x: x[0])
            optim_result = [x[1] for x in optim_result][:int(0.1 * cycle_range)]
            
            if verbose == 2:
                print('First approximation is obtained')

            for start_w_new in optim_result:

                if class_num == 1:
                    res = minimize(self.compute_train_loss_class_1, x0=start_w_new, method='Nelder-Mead', \
                             options={'disp': False, 'adaptive':True, 'xatol':1})
                elif class_num == 0:
                    res = minimize(self.compute_train_loss_class_0, x0=start_w_new, method='Nelder-Mead', \
                             options={'disp': False, 'adaptive':True, 'xatol':1})

                for k in range(2):
                    if class_num == 1:
                        res = minimize(self.compute_train_loss_class_1, x0=res.x, method='Nelder-Mead', \
                                 options={'disp': False, 'adaptive':True})
                    elif class_num == 0:
                        res = minimize(self.compute_train_loss_class_0, x0=res.x, method='Nelder-Mead', \
                                 options={'disp': False, 'adaptive':True})

                optim_result_more_precise += [[res.fun, res.x]]
                
            optim_result_more_precise = pd.DataFrame(optim_result_more_precise)
            optim_result_more_precise = optim_result_more_precise.sort_values(0).head(1)
            hyperplane_coefficients = optim_result_more_precise[1].values[0]
            min_loss_func = optim_result_more_precise[0].values[0]
        
        elif optim_method == 'differential_evolution':
            
            bounds = []
            for j in range(X_train.shape[1]):
                bounds.append((-10, 10))
                
            if class_num == 1:
                res = differential_evolution(self.compute_train_loss_class_1, bounds=bounds)
            
            if class_num == 0:
                res = differential_evolution(self.compute_train_loss_class_0, bounds=bounds)
            
            hyperplane_coefficients = res.x
            min_loss_func = res.fun
            
        else:
            
            for i in range(cycle_range):

                start_w = np.array((np.random.rand(X_train.shape[1]) - 0.5) * c)

                if class_num == 1:

                        res = minimize(self.compute_train_loss_class_1, x0=start_w, method=optim_method)

                elif class_num == 0:

                        res = minimize(self.compute_train_loss_class_0, x0=start_w, method=optim_method)

                optim_result.append([res.fun, res.x])


            if optim_result == []:
                raise Exception('We\'ve not get any satisfying first approximation')

            optim_result.sort(key=lambda x: x[0])
            optim_result = [x[1] for x in optim_result][:int(0.1 * cycle_range)]
            
            if verbose == 2:
                print('First approximation is obtained')

            for start_w_new in optim_result:

                if class_num == 1:
                    res = minimize(self.compute_train_loss_class_1, x0=start_w_new, method=optim_method)
                elif class_num == 0:
                    res = minimize(self.compute_train_loss_class_0, x0=start_w_new, method=optim_method)

                for k in range(2):
                    if class_num == 1:
                        res = minimize(self.compute_train_loss_class_1, x0=res.x, method=optim_method)
                    elif class_num == 0:
                        res = minimize(self.compute_train_loss_class_0, x0=res.x, method=optim_method)

                optim_result_more_precise += [[res.fun, res.x]]
                
            optim_result_more_precise = pd.DataFrame(optim_result_more_precise)
            optim_result_more_precise = optim_result_more_precise.sort_values(0).head(1)
            hyperplane_coefficients = optim_result_more_precise[1].values[0]
            min_loss_func = optim_result_more_precise[0].values[0]
        
        if verbose == 2:
            print('The minimum of the loss function: {0}'.format(min_loss_func))
        if verbose == 2:
            print('Time taken for optimization: {0}'.format(time.time() - start_time))

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
    
    
    def fit(self, X, y, optim_method='Nelder-Mead', cycle_range=100, disp=False, adaptive=True, maxiter=10, xatol=0.3, \
           verbose=0):
        """
            Fits the algorithm on the train sample 
            
            :param X: матрица признаков, обучающая выборка
            :param y: вектор истинных значений целевой переменной обучающей выборки
            :param optim_method: метод оптимизации: ['Nelder-Mead', 'differential_evolution', 'BFGS', 'CG', 'SLSQP', \
                                                    'COBYLA', 'TNC']
            Параметры оптимизации с помощью алгоритма Нелдера-Мида:
                :param disp: bool: печать сообщения о сходимости
                :param adaptive: bool: адаптация параметров алгоритма для размерности задачи (полезно при больших размерностях)
                :param maxiter: максимально допустимое количество итераций при оптимизации
                :param xatol: абсолютная ошибка на оптимальных точках между итерациями, приемлемая для сходимости
            :param verbose: подробный вывод описания обучения: [0, 1, 2]:
                0 - не печатать ничего
                1 - печатать общее время обучения
                2 - подробный вывод информации о процессе обучения
            :returns: -, но на выходе обученная моделька
        """
        
        start_of_fit_time = time.time()
        
        X_new = self.expand(X)
        
        self.X_train = X_new
        self.y_train = y
        self.optim_params = {'cycle_range': cycle_range, 'disp': disp, 'adaptive': adaptive, \
                             'maxiter': maxiter, 'xatol': xatol}
        
        hp_num = 1
        
        while self.X_train.shape[0] > 2 * self.N:
            
            if verbose == 2:
                print('Making hyperplane number {}'.format(hp_num))
                print('X_train.shape[0] = {}'.format(self.X_train.shape[0]))
            
            k = 10
            
            while True:
                
                if verbose == 2:
                    print('k = {}'.format(k))
                
                self.L = 2 ** k
                
                if verbose == 2:
                    print('L = {}'.format(self.L))
                
                if verbose == 2:
                    print('Optimizing hyperplane for class 1')
                
                hp_weights_class_1 = self.make_hyperplane(class_num=1, X_train=self.X_train, \
                                                     optim_method=optim_method, \
                                                     c=0.1, cycle_range=self.optim_params['cycle_range'], \
                                                     disp=self.optim_params['disp'], \
                                                     adaptive=self.optim_params['adaptive'], \
                                                     maxiter=self.optim_params['maxiter'], \
                                                     xatol=self.optim_params['xatol'], \
                                                     verbose=verbose)
                
                if verbose == 2:
                    print('Optimizing hyperplane for class 0')
                
                hp_weights_class_0 = self.make_hyperplane(class_num=0, X_train=self.X_train, \
                                                     optim_method=optim_method, \
                                                     c=0.1, cycle_range=self.optim_params['cycle_range'], \
                                                     disp=self.optim_params['disp'], \
                                                     adaptive=self.optim_params['adaptive'], \
                                                     maxiter=self.optim_params['maxiter'], \
                                                     xatol=self.optim_params['xatol'], \
                                                     verbose=verbose)
                
                cut_1 = self.cutter(self.X_train, hp_weights_class_1)
                cut_0 = self.cutter(self.X_train, hp_weights_class_0)
                X_1 = self.X_train[cut_1 == 1]
                y_1 = self.y_train[cut_1 == 1]
                y_1_rest = self.y_train[cut_1 == 0] # оставшиеся после отсечения сэмплы
                X_0 = self.X_train[cut_0 == 1]
                y_0 = self.y_train[cut_0 == 1]
                y_0_rest = self.y_train[cut_0 == 0] # оставшиеся после отсечения сэмплы
                
                if verbose == 2:
                    print('X_1.shape[0] = {}'.format(X_1.shape[0]))
                    print('X_0.shape[0] = {}'.format(X_0.shape[0]))
                
                if X_1.shape[0] < self.N and X_0.shape[0] < self.N:
                    
                    if verbose == 2:
                        print('Cutted data shape is not enough\n')
                    k -= 1
                    continue
                    
                else:
                    
                    if X_1.shape[0] >= X_0.shape[0]:
                        
                        proba = y_1.sum() / len(y_1)
                        proba_rest = y_1_rest.sum() / len(y_1_rest)
                        self.weights_hp[hp_num] = (1, hp_weights_class_1, proba, proba_rest)
                        
                        self.X_train = self.X_train[cut_1 == 0]
                        self.y_train = self.y_train[cut_1 == 0]
                        
                    elif X_1.shape[0] < X_0.shape[0]:
                        
                        proba = y_0.sum() / len(y_0)
                        proba_rest = y_0_rest.sum() / len(y_0_rest)
                        self.weights_hp[hp_num] = (0, hp_weights_class_0, proba, proba_rest)
                        
                        self.X_train = self.X_train[cut_0 == 0]
                        self.y_train = self.y_train[cut_0 == 0]
                        
                    hp_num += 1
                    if verbose == 2:
                        print()
                    break
                    
        end_of_fit_time = time.time()
        
        if verbose == 2 or verbose == 1:
            print('Time taken to fit the model: {0}'.format(end_of_fit_time - start_of_fit_time))
                

    def predict_proba(self, X):
        """
            Makes predict_proba for the sample
            
            :param X: матрица признаков, обучающая выборка
            :returns: предсказания вероятностей отнесения к классу 1
        """
        
        start_of_predict_time = time.time()
        
        X_new = self.expand(X)
        
        test_ = X_new.copy()
        
        if self.weights_hp == {}:
            raise Exception('Model is not fitted')
        
        predictions = pd.DataFrame({'proba': np.zeros(X.shape[0]), 'is_scored': False})
        
        for hp_num in self.weights_hp.keys():
            
            class_num, weights, proba, proba_rest = self.weights_hp[hp_num]
            
            cut = self.cutter(test_, weights)
            
            X_ = test_[cut == 1]
            test_ = test_[cut == 0]
            
            k = 0
            for i in range(predictions.shape[0]):
                
                if predictions.loc[i, 'is_scored'] == False:
                    if cut[k] == True:
                        predictions.loc[i, 'proba'] = proba
                        predictions.loc[i, 'is_scored'] = True
                    k += 1
            if hp_num == list(self.weights_hp.keys())[-1]:
                for i in range(predictions.shape[0]):
                    if predictions.loc[i, 'is_scored'] == False:
                        predictions.loc[i, 'proba'] = proba_rest
                        predictions.loc[i, 'is_scored'] = True
                        
        end_of_predict_time = time.time()
        print('Time taken to predict the targets: {0}'.format(end_of_predict_time - start_of_predict_time))
        
        return predictions['proba'].values
    
    def predict(self, X):
        """
            Makes predict_proba for the sample
            
            :param X: матрица признаков, обучающая выборка
            :returns: предсказания классов
        """
        
        proba = self.predict_proba(X)
        return np.array([round(x) for x in proba])