from tb import train
import sys

tags = sys.argv[1]
batch = sys.argv[2]
device = sys.argv[3]

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

if __name__ == '__main__':
    midpoint = len(treebanks)//2
    if batch == 'a':
        treebanks = treebanks[:midpoint]
    elif batch == 'b':
        treebanks = treebanks[midpoint:]

    for treebank in treebanks:
        train(treebank, tags, device)