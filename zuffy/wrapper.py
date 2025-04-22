"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
This module contains the zuffy Wrappers and supporting methods and functions.
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils._param_validation import StrOptions, Interval, Options, validate_params
import numbers # for scikit learn Interval


@validate_params( 
    {
    "fuzzy_X":      ["array-like"],
    "y":            ["array-like"],
    "n_iter":       [Interval(numbers.Integral, 1, None, closed="left")],
    "split_at":     [Interval(numbers.Real, 0, 1, closed="both")],
    "random_state": ["random_state"],
    }, 
    prefer_skip_nested_validation=True
)
class ZuffyFitIterator:
    '''
    ZuffyFitIterator documentation to be done.
    '''

    performance = None
    best_est = None
    best_score = None
    smallest_tree = None

    def __init__(self, fptgp, fuzzy_X, y, n_iter = 5, split_at=0.2, random_state=0):
        self.fptgp = fptgp

        fptgp._validate_params()

        self.fuzzy_X = fuzzy_X
        self.y = y
        self.n_iter = n_iter
        self.split_at = split_at
        self.random_state = random_state
        self.performIteration(fptgp, fuzzy_X, y)

    def performIteration(self, fptgp, fuzzy_X, y):
        '''
        performIteration documentation to be done.
        '''
        # now call the iter function to split and train the dataset n_iter times

        best_score = -np.inf
        best_iter  = -np.inf
        smallest_tree  = np.inf
        iter_perf = []
        sum_scores = 0
        for iter in range(self.n_iter):
                #self.verbose_out(f"{iter=}")
                iter_time = time.time()
                if self.random_state != None:
                    rs = self.random_state + iter
                else:
                    rs = self.random_state
                score, est_gp, class_scores = self.ZuffyFitJob(fptgp, fuzzy_X, y, split_at=self.split_at, random_state=rs)
                sum_scores += score
                self.verbose_out(f"{class_scores=}")
                # calculate the size of the model
                len_progs = 0
                for e in est_gp.estimators_:
                        len_progs += len(e._program.program)
                self.verbose_out(f'Tree size is {len_progs}')
                iter_perf.append([score, len_progs, class_scores])
                if (score > best_score) or ((score == best_score) and (len_progs < smallest_tree) ):
                    best_iter = iter
                    best_est = est_gp
                    best_score = score
                    smallest_tree = len_progs
                    self.verbose_out(f'\aNew leader with score {score} and size {len_progs}')
                iter_dur = round(time.time() - iter_time,0)
                avg_score = sum_scores/(iter +1)
                self.verbose_out(f'Duration of iteration #{iter} is {iter_dur}s # Best so far: {best_score:.5f}/{smallest_tree}  Avg: {avg_score:.5f}')

        self.verbose_out(f"Finished iterating - {best_iter=}")
        self.best_est = best_est
        self.best_score = best_score
        self.iter_perf = iter_perf
        self.best_iter = best_iter
        self.smallest_tree = smallest_tree
        #return best_est, best_score, iter_perf, best_iter

    @validate_params( 
        {
        "fuzzy_X":      ["array-like"],
        "y":            ["array-like"], # dict, list of dicts, "balanced", or None
        "n_iter":       [Interval(numbers.Integral, 1, None, closed="left")],
        "split_at":     [Interval(numbers.Real, 0, 1, closed="both")],
        "random_state": ["random_state"],
        }, 
        prefer_skip_nested_validation=True
    )
    def ZuffyFitJob(self, fptgp, fuzzy_X, y, split_at = 0.25, random_state=0):
        '''
        ZuffyFitJob documentation to be done.
        '''
        X_train, X_test, y_train, y_test = train_test_split(fuzzy_X, y, test_size=split_at, random_state=random_state)
        res   = fptgp.fit(X_train, y_train)
        score = res.score(X_test,y_test)
        self.verbose_out(f'Multi score: {score:.8f}')
        '''
        Can we now test each branch for each class and score them individually so that we can combine the best branches?
        '''
        #predictions = fptgp.predict(X_test) # which should we use res or fptgp?
        predictions = res.predict(X_test)
        class_scores = {}
        sum_scores = 0
        score_cnt  = 0
        if 1:
            #for cls in np.unique(y):
            for cls in fptgp.classes_:
                cls_idx = y_test == cls
                if len(y_test[cls_idx]) > 0:
                    class_score = accuracy_score(y_test[cls_idx], predictions[cls_idx])
                    self.verbose_out(f'Score for class {cls} is {class_score}')
                    class_scores[cls] = class_score
                    sum_scores += class_score # if not np.isnan(class_score) else 0
                    score_cnt += 1
                    #if score > best_models[cls]['score']:
                    #    best_models[cls] = {'model': clf, 'score': score}
                else:
                    self.verbose_out(f'class {cls} is not present in this split.')
            #avg_score = round(sum_scores/len(class_scores),5)
            avg_score = round(sum_scores/score_cnt,5)
            #self.verbose_out(f"{avg_score=}\nDiff: {round(avg_score - score,5)=}")

        if 0:
            print('Final(?) Population is:')
            for i,p in enumerate(res.estimators_[0]._programs[-1]):
                print(i,p)
            print("\n".join([str(p.parents) for p in res.estimators_[0]._programs[-1] ]))

        return score, res, class_scores
    
    def verbose_out(self, *msg: str) -> None:
        '''
        Display an informational message if the model is in verbose mode.
        '''
        if self.fptgp.verbose:
            for m in msg:
                print(m)

    def getBestEstimator(self):
        '''
        getBestEstimator documentation to be done.
        '''
        return self.best_est
    
    def getBestScore(self):
        '''
        getBestScore documentation to be done.
        '''
        return self.best_score
    
    def getPerformance(self):
        return self.iter_perf
    
    def getBestIter(self):
        return self.best_iter
    
    def getSmallestTree(self):
        return self.smallest_tree
    
    def getBestClass(self):
        best_class_score = -1
        best_class_performances = self.iter_perf[self.best_iter][2]
        for class_name, class_score in best_class_performances.items():
            if class_score > best_class_score:
                best_class_score = class_score
                best_class_name = class_name
        self.verbose_out(f'Best class score is {class_score} and indicates our tree is for when Target={best_class_name}')
        return best_class_name