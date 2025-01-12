import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def FPTGP_fit_job(fptgp, fuzzy_X, y, split_at = 0.25):
    X_train, X_test, y_train, y_test = train_test_split(fuzzy_X, y, test_size=split_at, random_state=22)
    res   = fptgp.fit(X_train, y_train)
    score = res.score(X_test,y_test)
    print('Multi score',score)
    '''
    Can we now test each branch for each class and score them individually so that we can combine the best branches?
    '''
    predictions = fptgp.predict(X_test)
    class_scores = {}
    sum_scores = 0
    if 1:
        for cls in np.unique(y):
            cls_idx = y_test == cls
            class_score = accuracy_score(y_test[cls_idx], predictions[cls_idx])
            print(f'Score for class {cls} is {class_score}')
            class_scores[cls] = class_score
            sum_scores += class_score
            #if score > best_models[cls]['score']:
            #    best_models[cls] = {'model': clf, 'score': score}
        avg_score = round(sum_scores/len(class_scores),5)
        print(f"{avg_score=}")
        print(f"Diff: {round(avg_score - score,5)=}")

    if 0:
        print('Final(?) Population is:')
        for i,p in enumerate(res.estimators_[0]._programs[-1]):
            print(i,p)
        print("\n".join([str(p.parents) for p in res.estimators_[0]._programs[-1] ]))

    return score, res, class_scores

def FPTGP_fit_iterator(fptgp, fuzzy_X, y, n_iter = 10, split_at=0.2):
    # now call the iter function to split and train the dataset n_iter times

    best_score = -np.inf
    best_iter  = -np.inf
    smallest_tree  = np.inf
    print('*****************************************************************************************')
    print('*****************************************************************************************')
    print('*****************************************************************************************')
    iter_perf = []
    for iter in range(n_iter):
            print(f"{iter=}")
            iter_time = time.time()
            score, est_gp, class_scores = FPTGP_fit_job(fptgp, fuzzy_X, y, split_at=split_at)
            print(f"{class_scores=}")
            # calculate the size of the model
            len_progs = 0
            for e in est_gp.estimators_:
                    len_progs += len(e._program.program)
            print(f'Tree size is {len_progs}')
            iter_perf.append([round(score,5), len_progs, class_scores])
            if (score > best_score) or ((score == best_score) and (len_progs < smallest_tree) ):
                best_iter = iter
                best_est = est_gp
                best_score = score
                smallest_tree = len_progs
                print(f'\aNew leader with score {score} and size {len_progs}')
            iter_dur = round(time.time() - iter_time,0)
            print(f'Duration of iteration #{iter} is {iter_dur}s #')

    print(f"{best_iter=}")
    return best_est, best_score, iter_perf
