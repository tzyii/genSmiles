import os
import pickle, yaml
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.DataStructs as DataStructs
from tokens import Tokenizer, getTokenizer

# suppress rdkit error
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def __load_config(fyaml='config.yaml'):
    with open(fyaml, 'r') as f:
        config = yaml.full_load(f)
    config['working_dir'] = os.path.abspath(config['working_dir'])
    config['fname_dataset'] = os.path.join(config['working_dir'], config['fname_dataset'])
    config['fname_fps'] = os.path.join(config['working_dir'], config['fname_fps'])
    config['fname_tokenizer'] = os.path.join(config['working_dir'], config['fname_tokenizer'])
    config['fname_rnn_parameters'] = os.path.join(config['working_dir'], config['fname_rnn_parameters'])
    config['fname_vae_encoder_parameters'] = os.path.join(config['working_dir'], config['fname_vae_encoder_parameters'])
    config['fname_vae_decoder_parameters'] = os.path.join(config['working_dir'], config['fname_vae_decoder_parameters'])
    config['sampled_dir'] = os.path.join(config['working_dir'], config['sampled_dir'])
    config['rnn_param']['maxLength'] = config['maxLength']
    config['vae_param']['maxLength'] = config['maxLength']
    return config

config = __load_config()

def mkdir_multi(path_str):
    if os.path.isdir(path_str) or path_str == '':
        return
    else:
        dir_namme = os.path.dirname(path_str)
        mkdir_multi(dir_namme)
        os.mkdir(path_str)

def get_tokenizer():
    if os.path.exists(config['fname_tokenizer']):
        print('read tokenizer from pickle file "%s"' % (config['fname_tokenizer']))
        with open(config['fname_tokenizer'], 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = getTokenizer(config['fname_dataset'], handleBraces=True)
        with open(config['fname_tokenizer'], 'wb') as f:
            pickle.dump(tokenizer, f)
    config['rnn_param']['num_embeddings'] = tokenizer.getTokensSize()
    config['rnn_param']['padding_idx'] = tokenizer.getTokensNum('<pad>')
    config['vae_param']['num_vocabs'] = tokenizer.getTokensSize() - 2
    return tokenizer

def isValidSmiles(smiles):
    if smiles == '':
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def getSimpleScore(smiles):
    if smiles == '':
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    else:
        for ma in mol.GetAtoms():
            if ma.GetAtomicNum() not in (1, 6, 7, 8, 9, 17):
                return -1
        return 1
        #return max((Descriptors.MolWt(mol) / 300.0 - 1.0)**2 + 1, 1.0e-2)

class SimilEvaluator(object):
    def __init__(self, smi_fname, pkl_fname, maxLength):
        self.maxLength = maxLength
        if os.path.isfile(pkl_fname):
            with open(pkl_fname, 'rb') as f:
                self.fps = pickle.load(f)
        else:
            self.fps = []
            with open(smi_fname, 'r') as f:
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    mol = Chem.MolFromSmiles(line.strip())
                    if mol is not None:
                        self.fps.append(self.genFP(mol))
            with open(pkl_fname, 'wb') as f:
                pickle.dump(self.fps, f)

    @staticmethod
    def genFP(mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useFeatures=True)

    def getSimil(self, smiles):
        if len(smiles) < 1 or len(smiles) >= self.maxLength:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = self.genFP(mol)
        simil = max(DataStructs.BulkTanimotoSimilarity(fp, self.fps))
        return self.rescaleScores(simil)
    
    def rescaleScores(self, simil):
        if simil < 0.2:
            return -1
        elif simil > 0.7:
            return 1
        else:
            return (simil - 0.2) * 4 - 1.0

