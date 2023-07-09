import utils, os, pickle
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from tqdm import tqdm

chembl_smiles = []
chembl_fps = []
with open(utils.config['fname_dataset'], 'r') as f:
    chembl_smiles = set([sm.strip() for sm in f.readlines()])
with open(utils.config['fname_fps'], 'rb') as f:
    chembl_fps = pickle.load(f)

def analysis(fname, train_data=False):
    print('analysis "%s"' % (fname,))
    simil, weight = [], []
    if not train_data:
        with open(fname, 'r') as f:
            reinforce_smiles = [sm.strip() for sm in f.readlines()]
            reinforce_smiles = [sm for sm in reinforce_smiles if utils.isValidSmiles(sm)]
            unique_reinforce_smiles = set(reinforce_smiles)
            new_reinforce_smiles = unique_reinforce_smiles.difference(chembl_smiles)
    else:
        new_reinforce_smiles = chembl_smiles
    for sm in tqdm(new_reinforce_smiles):
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            if not train_data:
                fp = utils.SimilEvaluator.genFP(mol)
                simil.append(max(BulkTanimotoSimilarity(fp, chembl_fps)))
            weight.append(Descriptors.MolWt(mol))
    simil, weight = np.array(simil), np.array(weight)
    if not train_data:
        print("%s: valid= %.1f unique= %.1f new= %.1f simil_mean= %.3f simil_std= %.3f weight_mean= %6.1f weight_std= %6.1f" % (fname, len(reinforce_smiles) / 100, len(unique_reinforce_smiles) / 100, len(new_reinforce_smiles) / 100, simil.mean(), simil.std(), weight.mean(), weight.std()))
    else:
        print("%s: weight_mean= %6.1f weight_std= %6.1f" % (fname, weight.mean(), weight.std()))

analysis(utils.config['fname_dataset'], train_data=True)
for fname in ('rnn.smi', 'vae.smi', 'reinforce.smi', 'reinvent.smi'):
    fname = os.path.join(utils.config['sampled_dir'], fname)
    analysis(fname)
