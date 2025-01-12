'''
Functions to handle the display of a FPT
'''
# Author: Peter O'Mahony <nucleus@mahoonium.com>
#
# License: BSD 3 clause


import random
import time
import matplotlib.pyplot as plt
import pandas as pd
#import graphviz
from gplearn.functions import _Function

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance


class ObjectColour:
    '''
    This class can be used for managing the colour of an object to ensure that
    each instance of that object uses the same colour and new instances are given
    a new, previously unused, colour.  It applies to Features and Operators.
    '''

    def_operator_colours = [ # default list of operator colours (pale pastels)
        '#ff999922',
        '#99ff9922',
        '#9999ff22',
        '#99ffff22',
        '#ff99ff22',
        '#ffff9922',
        ]
    
    def_feature_colours = [ # default list of feature colours (strong)
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
        ]

    def __init__(self, colour_list=None):
        self.object_colours  = colour_list
        self.used_colours    = {}

    def getColour(self, object_name):
        cmap = self.object_colours
        if object_name in self.used_colours:
            return cmap[self.used_colours[object_name]]
        else:
            next_colour_id = len(self.used_colours) % len(cmap) # wrap around if at end of the list
            self.used_colours[object_name] = next_colour_id
            return cmap[next_colour_id]


class FeatureColour(ObjectColour):

    def __init__(self, colour_list=None):
        if colour_list == None:
            colour_list = self.def_feature_colours
        else:
            colour_list.extend(self.def_feature_colours)
        super().__init__(colour_list)


class OperatorColour(ObjectColour):

    def __init__(self, colour_list=None):
        if colour_list == None:
            colour_list = self.def_operator_colours
        else:
            colour_list.extend(self.def_operator_colours)
        super().__init__(colour_list)


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

def output_node(i, node, feature_names, FeatureColourx, imp_feat=None):
    if isinstance(node, int):
        if feature_names is None:
            feature_name = 'X%s' % node
        else:
            feature_name = feature_names[node]
        fill = FeatureColourx.getColour(feature_name)
        if feature_name in imp_feat:
            f_mean = round(imp_feat[feature_name][0],3)
            f_std = round(imp_feat[feature_name][1],3)
            f_rank = imp_feat[feature_name][2]
            extra = f"{f_rank}: {f_mean} +/- {f_std}"
        else:
            extra = ""
        #output += ('%d [label="%s", color="black", fillcolor="%s", shape=none, image="fpt_node_1x5.png"] ;\n'

        #output += ('%d [label="%s", color="black", fillcolor="%s", shape=rectangle, style=filled] ;\n'
        #            % (i, addCR(feature_name), fill))

        #output = ('%d [label=<<table border="0" cellborder="0"><tr><td>%s</td></tr><tr><td border="1" align="left" fixedsize="true" width="20" height="25"><img src="img_50_50.png"></td><td>%s</td></tr></table>>, color="black", fillcolor="%s", shape=rectangle] ;\n'
        #output = ('%d [label=<<table border="0" cellborder="0"><tr><td>%s</td></tr><tr><td bgcolor="red">%s</td></tr></table>>, color="black", fillcolor="%s", shape=rectangle] ;\n'
        html_cell = f'<tr><td bgcolor="white" color="red">{extra}</td></tr>' if extra else ''
            
        output = ('%d [label=<\
                  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgcolor="%s">\
                  <tr><td>%s</td></tr>\
                  %s\
                  </table>>,\
                  color="black", shape=none] ;\n'
                    % (i, fill, addCR(feature_name), html_cell))
    else:
        output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                    % (i, node, fill))
    return output

def export_graphviz(program, feature_names=None, fade_nodes=None, start=0, fillcolor='green', operator_col_fn=None, feature_col_fn=None, imp_feat=None):
    '''
    Returns a string, Graphviz script for visualizing the program.

    Parameters
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
    
    terminals = []
    output = ''
    # Initialise the colour switchers
    operatorColourx = operator_col_fn # OperatorColour()
    FeatureColourx  = feature_col_fn # FeatureColour()

    for i, node in enumerate(program.program):
        i = i + start
        fill = fillcolor
        if isinstance(node, _Function): # _Function):
            terminals.append([node.arity, i])
            fill = operatorColourx.getColour(node.name)
            output += ('%d [label="%s", style=filled, fillcolor="%s"] ;\n'
                        % (i, node.name, fill))
        else:
            output += output_node(i, node, feature_names, FeatureColourx, imp_feat)
            if i == start:
                # A degenerative program of only one node
                return output #+ '}'
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

def graphviz_tree(est_list, target_names=None, feature_names=None, tree_name=None, bgcolor='white', imp_feat=None):

    # Initialise the colour switchers
    operatorColourx = OperatorColour()
    FeatureColourx  = FeatureColour()

    # need to ensure no more than scale nodes
    scale = 1000
    wta_id = (len(est_list) + 1) * scale
    wta_edges = ''
    wta_ports = ''
    out = 'digraph G {\n'
    out += f'bgcolor="{bgcolor}"\n'
    out += f'fontname="Helvetica"\n'
    out += f'fontsize="22"\n'    # title size
    out += f'node [fontname="Helvetica"]\n'
    #out += f'node [shape=none label=""]\n'
    #out += f'node [imagescale=true]\n'
    #out += f'n1 [image="fpt.png"]\n'
    if tree_name:
        out += f'label="{tree_name}"\n'
        out += f'labelloc  =  t\n'
    for idx, e in enumerate(est_list):
        #dot_data.append(export_graphviz(e._program, start=idx*100))
        out += export_graphviz(e._program, start=idx*scale, fillcolor='#{:06x}'.format(random.randint(0, 0xFFFFFF)), feature_names=feature_names, operator_col_fn=operatorColourx, feature_col_fn=FeatureColourx,imp_feat=imp_feat )
        #wta_edges += '%d:%s -> %d [label="%s"];\n' % (wta_id, 'port_' + str(idx), idx*scale, target_names[idx]) # 'class_' + str(idx))
        wta_edges += '%d:%s -> %d;\n' % (wta_id, 'port_' + str(idx), idx*scale) # 'class_' + str(idx))
        score = e._program.raw_fitness_
        wta_ports += "<td port='port_%d'>%s</td>" % (idx, str(target_names[idx]) + f" ({score:3.3f})" ) # 'class_' + str(idx))

    #out += ('%d [label="%s", color="%s", shape=record, style=filled, width=4] ;\n'
    #            % (wta_id, 'WTA', '#ffcc33'))
    out += ('%d [label=%s, color="%s", shape=plaintext, width=4, fontname="Helvetica"] ;\n'
                #% (wta_id, "<<table border='1' cellborder='1'><tr><td colspan='3'>WTA</td></tr><tr><td port='port_one'>First port</td><td port='port_two'>Second port</td><td port='port_three'>Third port</td></tr></table>>", '#ffcc33'))
                % (wta_id, f"<<table border='1' cellborder='1' bgcolor='grey'><tr><td colspan='{len(est_list)}'>WTA</td></tr><tr>{wta_ports}</tr></table>>", 'black'))
    out += wta_edges
    out += '}'
    #print(out)
    return out

def plot_evolution(model, target_classes, iter_perf, analyis_filename):
    hei=len(model.estimators_)
    wid=5
    fig = plt.figure(figsize=(11, 2.5 * hei))
    gidx = 1
    for idx, cls in enumerate(model.estimators_):
        print('Model Class#', target_classes[idx])

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title(f'Class: {target_classes[idx]}\nAvg Len ' + '  final: ' + str(round(cls.run_details_['average_length'][-1],4)))
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['average_length'], color='tab:blue')
        ax.plot(cls.run_details_['generation'], cls.run_details_['best_length'], color='tab:orange')
        #plt.show()

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title('Best Len ' + '\n' + str(round(cls.run_details_['best_length'][-1],4)))
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['best_length'], color='tab:orange')

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title('Best Fitness\n(smaller better)' + '\n' + str(round(cls.run_details_['best_fitness'][-1],4)))
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['best_fitness'], color='tab:green')
        ax.plot(cls.run_details_['generation'], cls.run_details_['average_fitness'], color='tab:purple')

        ax = fig.add_subplot(hei, wid, gidx)
        gidx += 1
        ax.set_title('Avg Fitness ' + '\n' + str(round(cls.run_details_['average_fitness'][-1],4)))
        ax.plot(cls.run_details_['generation'], cls.run_details_['average_fitness'], color='tab:purple')

        ax = fig.add_subplot(hei, wid, gidx)
        ax.set_title('Generation Duration' + '\n' + str(round(cls.run_details_['generation_time'][-1],4)))
        gidx += 1
        ax.plot(cls.run_details_['generation'], cls.run_details_['generation_time'], color='#ffcc33')


    plt.tight_layout()
    plt.savefig(analyis_filename)



def show_feature_importance(reg, X_test, y_test, features, importance_filename): # https://github.com/huangyiru123/symbolic-regression-and-classfier-based-on-NLP/blob/main/%E7%AC%A6%E5%8F%B7%E5%88%86%E7%B1%BBNLP.ipynb
    # Get feature-importance scores
    rept = 20
    #rept = 3
    start_time = time.time()
    result = permutation_importance(reg, X_test, y_test, n_repeats=rept) #, n_jobs=3) #, random_state=0)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    # https://scikit-learn.org/stable/modules/permutation_importance.html
    print('*** IMPORTANCES ***')
    imp_feat = {}
    imp_graph_name = []
    imp_graph_val = []
    rank = 1
    for i in result.importances_mean.argsort()[::-1]:
        #if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
        if result.importances_mean[i] != 0 or result.importances_std[i] !=0:
            #imp_feat.append([features[i], result.importances_mean[i], result.importances_std[i]])
            imp_feat[features[i]] = [result.importances_mean[i], result.importances_std[i], rank]
            rank += 1
            imp_graph_name.append(features[i])
            imp_graph_val.append(result.importances_mean[i])
            print(f"{features[i]:<40}"
                f"{result.importances_mean[i]:.3f}"
                f" +/- {result.importances_std[i]:.3f}")
            
    # --- plot top 10
    #tree_importances = pd.Series(result.importances_mean.argsort()[::-10], index=features)
    top_ten = sorted(enumerate(result.importances_mean), key=lambda x: x[1], reverse=True)[:10]
    tree_importances = pd.Series(top_ten, index=[features[i] if j>0 else None for i,j in top_ten])

    # Plot permutation feature importances
    fig, ax = plt.subplots()
    #tree_importances.plot.bar(yerr=[result.importances_std[i]  for i in top_ten ], ax=ax, color='blue')
    #tree_importances.plot.bar(yerr=[result.importances_mean[i] for i in top_ten ], ax=ax, color='green')
    plt.bar(imp_graph_name, imp_graph_val, color='#ffcc33')
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("TODO: Explain 'Mean accuracy decrease'")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='small')
    fig.tight_layout()
    plt.savefig(importance_filename)
    return imp_feat
    #plt.show()
    
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
    return imp_feat
    '''

def do_model_dt(X, y, features, outfilename):
    est_dt = DecisionTreeRegressor(
                max_leaf_nodes=         10,
                #verbose=                1,
                #n_jobs=                 3,
                random_state=           221
                )
    est_dt.fit(X, y)
    plt.figure(figsize=(16, 26))  # Width: 10 inches, Height: 6 inches  
    tree.plot_tree(est_dt, feature_names=features, filled=True, rounded=True,fontsize=10)
    plt.title('Decision Tree version of our FPT')  
    #plt.tight_layout()
    plt.savefig(outfilename)
    return est_dt

