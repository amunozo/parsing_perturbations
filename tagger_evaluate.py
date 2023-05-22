import sys
from tagger import evaluate   
import numpy as np


if __name__ == '__main__':
    tags = sys.argv[1]
    if tags == 'xpos':
        treebanks = [
            'UD_Afrikaans-AfriBooms', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_German-GSD',
            'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT',
            'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES'
        ]
    elif tags == 'feats':
        treebanks = [
            'UD_Afrikaans-AfriBooms', 'UD_Basque-BDT', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_German-GSD',
            'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Hungarian-Szeged',
            'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
        ]
    else:
        treebanks = [
            'UD_Afrikaans-AfriBooms', 'UD_Basque-BDT', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_German-GSD',
            'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT', 'UD_Hungarian-Szeged',
            'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
        ]

    acc_list = []
    dict_acc = {}
    for treebank in treebanks:
        acc = evaluate(treebank, tags)
        dict_acc[treebank] = acc
        if acc:
            acc_list.append(acc)

    # print the average accuracy
    print("Mean {}".format(np.mean(acc_list)))
    print(dict_acc)
