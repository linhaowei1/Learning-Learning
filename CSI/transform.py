import nlpaug.augmenter.word as naw
import random
augAnt = naw.AntonymAug(aug_p = 1.0)
augSyn = naw.SynonymAug(aug_src='wordnet', aug_p = 0.1) 
augSwap = naw.RandomWordAug(action = 'swap', aug_p = 0.5) # 90%的swap概率
augDel = naw.RandomWordAug(action = 'delete') # 没有指定变换概率，默认为30%
augMask = naw.RandomWordAug(action = 'substitute', target_words = ['<mask>'], aug_p = 0.1) # 没有指定变换概率，默认为30%

def SynTrans(data):
    return augSyn.augment(' '.join(data)).split(' ')
def SwapTrans(data):
    return augSwap.augment(' '.join(data)).split(' ')
def DelTrans(data):
    return augDel.augment(' '.join(data)).split(' ')
def AntTrans(data):
    return augAnt.augment(' '.join(data)).split(' ')
def MaskTrans(data):
    return augMask.augment(' '.join(data)).split(' ')

def aug(data):
    return MaskTrans(SynTrans(data))
