from rdkit import Chem
import json
import argparse
import os, sys
import numpy as np
from autoencoder import autoencoder


def decode(latent_mols_file, output_smiles_file_path=None, message=''):
    print("BEGIN")
    print("Message: %s " % message)
    sys.stdout.flush()
    model = autoencoder.load_model('hetero')

    if output_smiles_file_path is None:
        output_smiles_file_path = os.path.join(os.path.dirname(latent_mols_file), 'decoded_smiles.smi')

    with open(latent_mols_file, 'r') as f:
        latent = json.load(f)

    invalids = 0
    batch_size = 256    # decoding batch size
    n = len(latent)

    with open(output_smiles_file_path, 'w') as smiles_file:
        for indx in range(0, n // batch_size):
            lat = np.array(latent[(indx) * 256:(indx + 1) * 256])
            if indx % 10 == 0:
                print("Batch [%d/%d] decoded, [Invalids: %s]" % (indx, n // batch_size + 1, invalids))
                sys.stdout.flush()
                smiles_file.flush()
            smiles, _ = model.predict_batch(lat, temp=0)
            for smi in smiles:

                smiles_file.write(smi + '\n')
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    invalids += 1
    print("Decoding completed.")
    print("Total: [%d] Fraction Valid: [0.%d]" % (n, (n - invalids) / n * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and train a model")

    parser.add_argument("--latent_mols_file", "-l", help="The path to a data file.", type=str, required=True)
    parser.add_argument("--output_smiles_file_path", "-o", help="Prefix to the folder to save output smiles.", type=str)
    parser.add_argument("--message", "-m", help="Message printed before training.", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    decode(**args)
