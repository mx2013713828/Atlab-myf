#coding:utf-8
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,average_precision_score
import os,json,argparse
import matplotlib.pyplot as plt
def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', type=str, help='pred result')
    parser.add_argument('gt_file', type=str, help='gt file')
    # parser.add_argument('--output', type=str, help='output save file ')
    args = parser.parse_args()
    return args
def get_gt(gt_file,test_name=None):
    #get gt dic
    #input:gt_list and test_name  output:gt dict
    gt_dic = {}
    if 'blade' in gt_file:
        with open(gt_file) as f:
            for row in f:
        #         print row
                try:
                    key = row.split(' ')[0].split('/')[-1]
                    label = row.split(' ')[1].strip('\n')
                    gt_dic[key] = label
        #             break
                except:
                    pass 
    elif 'qpulp' in gt_file:
        with open(gt_file) as f:
            for row in f:
        #         print row
                try:
                    key = row.split(',')[1].split('/')[-1]
                    label = row.split(',')[2].strip('\n')
                    gt_dic[key] = label
        #             break
                except:
                    pass
    return gt_dic

def get_pred(pred_file,gt_dic):
    #get pred and pred list 
    #y_gt,y_pred to compute pr
    #input: pred_file(str),gt_dic(dict)
    #output:y_gt,y_pred,
    # y_score(lists, have 3 score list ,for 3 class),y_gt_score(like y_score but gt list)
    y_gt0 = []
    y_gt1 = []
    y_score0=[]
    y_score1=[]
    y_gt2= []
    y_score2 = []
    y_pred=[]
    y_gt=[]
    # y_score = []
    y_score =[]
    y_gt_score = []
    JS = json.load(open(pred_file))
    for key in JS.keys():
        gt = gt_dic[key]
        if gt =='0':
            y_gt0.append(1)
            y_gt1.append(0)
            y_gt2.append(0)
        elif gt == '1':

            y_gt0.append(0)
            y_gt1.append(1)
            y_gt2.append(0)
        elif gt =='2':

            y_gt0.append(0)
            y_gt1.append(0)
            y_gt2.append(1)
        y_gt.append(int(gt))
        pred = JS[key]['Top-1 Index'][0]
        score_0 = JS[key]['Confidence'][0]
        score_1 = JS[key]['Confidence'][1]
        score_2 = JS[key]['Confidence'][2]
    #     y_score.append(float(JS[key]['Confidence'][pred]))
        y_score0.append(float(score_0))
    #     y_score.append()
        y_score1.append(float(score_1))
        y_score2.append(float(score_2))
        y_gt_score.append(y_gt0)
        y_gt_score.append(y_gt1)
        y_gt_score.append(y_gt2)
        
    #     y_score.append(y_score0)
    #     y_score.append(y_score1)
    #     y_score.append(y_score2)
    #     if pred =='nsfw':
    #         y_pred.append(0)
    #     elif pred =='normal':
    #         y_pred.append(1)
        y_pred.append(int(pred))
    y_gt_score.append(y_gt0)
    y_gt_score.append(y_gt1)
    y_gt_score.append(y_gt2)

    y_score.append(y_score0)
    y_score.append(y_score1)
    y_score.append(y_score2)
    return y_gt,y_pred,y_score,y_gt_score
def get_diff():
    #ruturn 100 pr lists
    return None
def merge():
    print('merge')
    #merge two results from different model
    return None

def compute_pr(pred,gt,throuh):
    #input : pred and gt list,and Threshold
    #output : recall,precision,acc
    tp = float(0)
    fp = float(0)
    fn = float(0)
    tn = float(0)
    for i in range(len(pred)):
        predict = pred[i]
        groundtruth = gt[i]
        if predict>throuh:
            if groundtruth ==1:
                tp = tp+1
            elif groundtruth ==0:
                fp = fp+1
        if predict<throuh:
            if groundtruth ==1:
                fn = fn+1
            elif groundtruth ==0:
                tn = tn+1
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    return recall,precision,accuracy
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def main():
    args = _init_()
    pred_file = args.pred_file
    gt_file = args.gt_file
    print (pred_file,gt_file)
    gt_dic = get_gt(gt_file)
    y_gt,y_pred,y_score,y_gt_score = get_pred(pred_file,gt_dic)
    classes = ['pulp','sexy','normal']
    # cm = conf:100:usion_matrix(y_gt, y_pred, labels=np.arange(len(classes)))
    p = precision_score(y_gt, y_pred, average=None)
    r = recall_score(y_gt, y_pred, average=None)
    # ap = average_precision_score(y_gt,y_pred)
    acc = accuracy_score(y_gt, y_pred)
    print('accuracy:', acc)

    for i in range(len(classes)):
        ap = average_precision_score(y_gt_score[i],y_score[i])
        print('%s precision:' % classes[i], p[i])
        print('%s recall:' % classes[i], r[i])
        print('%s ap:'%classes[i],ap)

    print('Top-1 error ',1-acc)
    # plot_confusion_matrix(cm, classes)

if __name__ == '__main__':
    args = _init_()
    main()