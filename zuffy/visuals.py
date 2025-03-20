"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
Functions to handle the display of a FPT
"""

import random
import time
import html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz
from gplearn.functions import _Function

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils._param_validation import StrOptions, Interval, Options, validate_params, HasMethods
import numbers # for scikit learn Interval

class ObjectColor:
    '''
    This class can be used for managing the color of an object to ensure that
    each instance of that object uses the same color and new instances are given
    a new, previously unused, color.  It applies to Features and Operators.
    '''

    def_operator_colors = [ # default list of operator colors (pale pastels)
        '#ff999922', '#99ff9922', '#9999ff22', '#99ffff22', '#ff99ff22', '#ffff9922' ]
    
    def_feature_colors = [ # default list of feature colors (strong)
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' ]

    def __init__(self, color_list=None):
        self.object_colors  = color_list
        self.used_colors    = {}

    def getColor(self, object_name):
        cmap = self.object_colors
        if object_name in self.used_colors:
            return cmap[self.used_colors[object_name]]
        else:
            next_color_id = len(self.used_colors) % len(cmap) # wrap around if at end of the list
            self.used_colors[object_name] = next_color_id
            return cmap[next_color_id]


class FeatureColor(ObjectColor):

    def __init__(self, color_list=None):
        if color_list == None:
            color_list = self.def_feature_colors
        else:
            color_list.extend(self.def_feature_colors)
        super().__init__(color_list)

class OperatorColor(ObjectColor):

    def __init__(self, color_list=None):
        if color_list == None:
            color_list = self.def_operator_colors
        else:
            color_list.extend(self.def_operator_colors)
        super().__init__(color_list)

def addCR(a_string):
    '''
    Replace a bar character with a carriage return.
    This is used to make feature names more presentable.

    Parameters
    ----------
    a_string : str
        A string of text which may contain a bar (|) character.

    Returns
    -------
        A string containing the result of replacing | with \n
    '''
    return a_string.replace("|","\n")

def add_importance(feature):
    if not isinstance(feature,list) or len(feature) != 3:
        raise ValueError(f"The Feature Importance dictionary does not contain three elements (mean, stdev, rank) for this feature {feature}.")
    
    f_mean, f_std, f_rank = list(feature)
    if not isinstance(f_mean,float):
        raise ValueError(f"The Feature Importance dictionary does not have a numeric value for the mean ({feature}).")
    
    f_mean = round(feature[0],3)
    f_std = round(feature[1],3)
    f_rank = feature[2]
    extra = f'<font point-size="14"><b>{f_rank}</b>: {round(f_mean,3)} &plusmn; {round(f_std,3)}</font>'
    return extra
        
def output_node(i, node, featureNames, FeatureColorx, impFeat=None):
    if isinstance(node, int):
        if featureNames is None:
            feature_name = 'X%s' % node
        else:
            feature_name = featureNames[node]

        fill = FeatureColorx.getColor(feature_name)

        extra = ""
        if impFeat and feature_name in impFeat:
            extra = add_importance(impFeat[feature_name])

        feature_name = feature_name.replace('|', '<br/>')

        html_cell = f'<tr><td bgColor="white" color="red">{extra}</td></tr>' if extra else ''
            
        output = ('%d [label=<\
                  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgColor="%s">\
                  <tr><td>%s</td></tr>\
                  %s\
                  </table>>,\
                  color="black", shape=none] ;\n'
                    % (i, fill, addCR(feature_name), html_cell))
    else:
        output += ('%d [label="%.3f", fillColor="%s"] ;\n'
                    % (i, node, fill))
    return output

def sanitize_names(features_names: list[object]) -> list[str]:
    if features_names is None:
        return None
    
    sanitised = []
    for f in features_names:
        sanitised.append(html.escape(str(f)))
    return sanitised

def export_graphviz(program, featureNames=None, fade_nodes=None, start=0, fillColor='green', operator_col_fn=OperatorColor(), feature_col_fn=FeatureColor(), impFeat=None):
    '''
    Returns a string, Graphviz script for visualizing the program.

    Parameters TO BE DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO BE DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO BE DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ----------
    fade_nodes : list, optional
        A list of node indices to fade out for showing which were removed
        during evolution.

    Returns
    -------
    output : string
        The Graphviz script to plot the tree representation of the program.

    '''
    if not hasattr(program, 'program'):
        raise ValueError('The program parameter does not have a program attribute')
    
    if featureNames and len(featureNames)>0 and len(featureNames) < program.n_features:
        raise ValueError(f'There are insufficient featureNames ({len(featureNames)}) supplied for this program (which expects {program.n_features}).')
    
    terminals = []
    output = ''
    # Initialise the color switchers
    operatorColorx = operator_col_fn
    FeatureColorx  = feature_col_fn

    for i, node in enumerate(program.program):
        i = i + start
        fill = fillColor
        if isinstance(node, _Function):
            terminals.append([node.arity, i])
            fill = operatorColorx.getColor(node.name)
            output += ('%d [label="%s", style=filled, fillColor="%s"] ;\n'
                        % (i, node.name, fill))
        else:
            output += output_node(i, node, featureNames, FeatureColorx, impFeat)
            if i == start:
                # A degenerative program of only one node
                return output
            terminals[-1][0] -= 1
            terminals[-1].append(i)
            while terminals[-1][0] == 0:
                output += '%d -> %d ;\n' % (terminals[-1][1],
                                            terminals[-1][-1])
                terminals[-1].pop()
                if len(terminals[-1]) == 2:
                    parent = terminals[-1][-1]
                    terminals.pop()
                    if not terminals:
                        return output #+ '}'
                    terminals[-1].append(parent)
                    terminals[-1][0] -= 1

    # We should never get here
    return None

@validate_params( 
    {
    "est_list":         [HasMethods("fit")], 
    "targetName":       [None, str],  # The name to use at the top of our tree for each classification value
    "featureNames":     [None, list],
    "treeName":         [None, str],
    "bgColor":          [None, str],
    "impFeat":          [None, dict],
    "outputFilename":   [None, str],
    "featColorList":    [None, list],
    "operColorList":    [None, list],
    }, 
    prefer_skip_nested_validation=True
)
def graphviz_tree(
            fptgp,
            targetFeatureName='Target',
            targetClassNames=None, # a list like ['sentosa',''...] or ['transported','not transported']
            featureNames=None,
            treeName=None,
            impFeat=None,
            outputFilename='zuffy_output',
            bgColor='white',
            rootBGColor='grey',
            rootText='WTA',
            featColorList=None,
            operColorList=None,
            showFitness=False):

    # Initialise the color switchers
    operatorColorx = OperatorColor(operColorList)
    FeatureColorx  = FeatureColor(featColorList)

    if not featureNames and hasattr(fptgp.estimator, 'feature_names'):
        featureNames = fptgp.estimator.feature_names

    featureNames = sanitize_names(featureNames)

    if 0:
        if targetNames is not None and len(list(targetNames))>0:  # this allows for both numpy array and list
            if len(targetNames) < len(fptgp.estimators_):
                raise ValueError(f'There are insufficient targetNames ({len(targetNames)}) supplied to represent each of the {len(fptgp.estimators_)} classes.')
            else:
                targetNames  = sanitize_names(targetNames)
        else:
            #targetNames  = ['Target_' + str(i) for i in range(len(est_list))]
            #targetNames  = np.unique(fptgp.estimators_[0].y_) 
            targetNames = list(fptgp.classes_)

    # We need to translate the target classes from their numeric equivlent into str if our classes are strings and we need those strings sent in as a param
    if targetClassNames is None:
        targetClassNames = list(fptgp.classes_)

    # need to ensure no more than scale nodes
    scale = 1000
    wta_id = (len(fptgp.estimators_) + 1) * scale
    wta_edges = ''
    wta_ports = ''
    out = 'digraph G {\n'
    out += f'bgcolor="{bgColor}"\n'
    out += f'fontname="Helvetica"\n'
    out += f'fontsize="22"\n'    # title size
    out += f'node [fontname="Helvetica"]\n'
    #out += f'node [shape=none label=""]\n'
    #out += f'node [imagescale=true]\n'
    #out += f'n1 [image="fpt.png"]\n'
    if treeName:
        out += f'label="{treeName}"\n'
        out += f'labelloc  =  t\n'
    for idx, e in enumerate(fptgp.estimators_):
        # This is the tree for class called targetClassNames[idx] 
        if not hasattr(e, "_program"):
            raise ValueError("The Classifier list is expected to have the _program attribute so that it can build the tree but is is missing.")

        out += export_graphviz(e._program, start=idx*scale, fillColor='#{:06x}'.format(random.randint(0, 0xFFFFFF)), featureNames=featureNames, operator_col_fn=operatorColorx, feature_col_fn=FeatureColorx,impFeat=impFeat )
        wta_edges += '%d:%s -> %d;\n' % (wta_id, 'port_' + str(idx), idx*scale) # 'class_' + str(idx))

        if showFitness:
            extra = f" ({e._program.raw_fitness_:3.3f})"
        else:
            extra = ""
        wta_ports += "<td port='port_%d'>%s</td>" % (idx, targetFeatureName + '=' + str(targetClassNames[idx]) + extra ) # 'class_' + str(idx))

    out += ('%d [label=%s, color="%s", shape=plaintext, width=4, fontname="Helvetica"] ;\n'
                % (wta_id, f"<<table border='1' cellborder='1' bgColor='{rootBGColor}'><tr><td colspan='{len(fptgp.estimators_)}'>{rootText}</td></tr><tr>{wta_ports}</tr></table>>", 'black'))
    out += wta_edges
    out += '}'
    graph = graphviz.Source(out)
    _ = graph.render(outputFilename, format='png', view=False, cleanup=True)
    return out, graph

@validate_params( 
    {
    "model":            [HasMethods("fit")],
    "target_classes":   [None, "array-like"],
    "outputFilename":   [None, str],
    "iter_perf":        [None, list],
    }, 
    prefer_skip_nested_validation=True
)
def plot_evolution(model, target_classes=None, iter_perf=None, outputFilename=None):

    if target_classes is not None:
        if len(target_classes) < len(model.estimators_):
            raise ValueError(f'There are insufficient targetNames ({len(target_classes)}) supplied to represent each of the {len(model.estimators_)} classes.')
        else:
            target_classes  = sanitize_names(target_classes)
    else:
        #target_classes  = ['Target_' + str(i) for i in range(len(model.estimators_))]
        target_classes  = model.classes_

    hei=len(model.estimators_)
    wid=3
    fig = plt.figure(figsize=(11, 2.5 * hei))
    gidx = 1
    for idx, cls in enumerate(model.estimators_):
        #print('Model Class#', target_classes[idx])

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title(f'Class: {target_classes[idx]}\nTree Length' + ' (Final Avg: ' + str(round(cls.run_details_['average_length'][-1],2)) + ')')
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['average_length'], color='tab:blue',label='Average')
        ax.plot(cls.run_details_['generation'], cls.run_details_['best_length'], color='tab:orange',label='Best')
        ax.legend()
        #plt.show()

        if 0:
            ax = fig.add_subplot(hei, wid, gidx)
            ax.set_title('Best Len ' + '\n' + str(round(cls.run_details_['best_length'][-1],4)))
            gidx += 1
            ax.plot(cls.run_details_['generation'], cls.run_details_['best_length'], color='tab:orange')

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title('Fitness (smaller is better)' + '\nFinal Best: ' + str(round(cls.run_details_['best_fitness'][-1],3)))
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['average_fitness'], color='tab:purple',label='Average')
        ax.plot(cls.run_details_['generation'], cls.run_details_['best_fitness'], color='tab:green',label='Best')
        ax.legend()

        if 0:
            ax = fig.add_subplot(hei, wid, gidx)
            gidx += 1
            ax.set_title('Avg Fitness ' + '\n' + str(round(cls.run_details_['average_fitness'][-1],4)))
            ax.plot(cls.run_details_['generation'], cls.run_details_['average_fitness'], color='tab:purple')

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title('Generation Duration' + '\nAverage: ' + str(round( sum(cls.run_details_['generation_time'])/len(cls.run_details_['generation_time']),4)))
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['generation_time'], color='#ffcc33')


    plt.tight_layout()
    if outputFilename:
        plt.savefig(outputFilename)
    else:
        plt.show()

@validate_params( 
    {
    "reg":              [HasMethods("fit")],
    "X_test":           ["array-like"],
    "y_test":           ["array-like"],
    "features":         [None, list],
    "outputFilename":   [None, str],
    }, 
    prefer_skip_nested_validation=True
)
def show_feature_importance(reg, X_test, y_test, features=None, n_jobs=None, n_repeats=20, outputFilename=None): # https://github.com/huangyiru123/symbolic-regression-and-classfier-based-on-NLP/blob/main/%E7%AC%A6%E5%8F%B7%E5%88%86%E7%B1%BBNLP.ipynb
    # Get feature-importance scores

    if not features and hasattr(reg.estimators_[0], 'feature_names'):
        features = reg.estimators_[0].feature_names

    start_time = time.time()
    result = permutation_importance(reg, X_test, y_test, n_repeats=n_repeats, n_jobs=n_jobs)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    # https://scikit-learn.org/stable/modules/permutation_importance.html
    print('*** IMPORTANCES ***')
    impFeat = {}
    imp_graph_name = []
    imp_graph_val = []
    rank = 1
    for i in result.importances_mean.argsort()[::-1]:
        if result.importances_mean[i] != 0 or result.importances_std[i] !=0:
            impFeat[features[i]] = [result.importances_mean[i], result.importances_std[i], rank]
            rank += 1
            imp_graph_name.append(features[i])
            imp_graph_val.append(result.importances_mean[i])
            print(f"{features[i]:<40}"
                f"{result.importances_mean[i]:.3f}"
                f" +/- {result.importances_std[i]:.3f}")
            
    # --- plot top 10
    top_ten = sorted(enumerate(result.importances_mean), key=lambda x: x[1], reverse=True)[:10]
    tree_importances = pd.Series(top_ten, index=[features[i] if j>0 else None for i,j in top_ten])

    # Plot permutation feature importances
    fig, ax = plt.subplots()
    plt.bar(imp_graph_name, imp_graph_val, color='#ffcc33')
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("TODO: Explain 'Mean accuracy decrease'")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='small')
    fig.tight_layout()
    if outputFilename:
        plt.savefig(outputFilename)
    else:
        plt.show()
    return impFeat
    
    '''
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = pom_plot_permutation_importance(reg, X_test, y_test, ax)
    ax.set_title("Permutation Importances on selected subset of features\n(test set)")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.ioff()
    plt.savefig(out_folder + "/feat_imp_box_" + str(iter))
    #plt.show()

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(result.importances_mean.reshape(1, -1), cmap='YlGnBu')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set tick labels
    ax.set_xticks(np.arange(len(result.importances_mean)))
    #tick_labs = [f"Feature {i+1}" for i in range(len(result.importances_mean))]
    tick_labs = [features[i] for i in range(len(result.importances_mean))]
    ax.set_xticklabels(tick_labs)
    ax.set_yticks([])

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Set plot title and show plot
    ax.set_title("Feature Importance Scores")
    outfilename = out_folder + "/feat_imp_mc_" + str(iter)
    plt.savefig(outfilename)
    print("Feature Importance Scores written to ",outfilename)
    plt.ioff()
    #plt.show(block=False)    
    #plt.show()
    return impFeat
    '''

def do_model_dt(X, y, features, outfilename):
    est_dt = DecisionTreeRegressor(
                max_leaf_nodes=         10,
                #verbose=                1,
                #n_jobs=                 3,
                random_state=           221
                )
    est_dt.fit(X, y)
    if 1:
        plt.figure(figsize=(10, 10))  # Width: 10 inches, Height: 6 inches   # adjust this depending on depth/width of tree?

    tree.plot_tree(est_dt, featureNames=features, filled=True, rounded=True,fontsize=6) #produces squashed trees
    tree.export_text(est_dt, featureNames=features)
    t = plt.title('Decision Tree version of our FPT')  
    print('export tree gives:\n',t)
    #plt.tight_layout()
    plt.savefig(outfilename)
    return est_dt

