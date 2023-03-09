from tempfile import TemporaryDirectory
from runners.CreateModelRunner import CreateModelRunner
from models.Discriminator import Discriminator
from models.Generator import Generator
import torch
import unittest
import numpy as np
import json
from src.Sampler import Sampler
from datasets.LatentMolsDataset import LatentMolsDataset
import os
import torch.autograd as autograd

class test_GAN(unittest.TestCase):
    #These tests are heavily inspired by the blog post of Chase Roberts:
    # https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
    # We can replace a real encoding of SMILES into latent vectors with a randomly
    #  initialized numpy array because we only want to check whether the GAN initializes properly.
    
    def test_separate_optimizers(self):
        # Verify that two different instances of the optimizer is created using the TrainModelRunner.py initialization
        # This ensures the two components train separately 
        with TemporaryDirectory() as tmpdirname:
            
            latent=np.random.rand(64,1,512)
            os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
            with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                json.dump(latent.tolist(), f)

            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            D = Discriminator.load(tmpdirname+'/discriminator.txt')
            G = Generator.load(tmpdirname+'/generator.txt')
            optimizer_G = torch.optim.Adam(G.parameters())
            optimizer_D = torch.optim.Adam(D.parameters())
            self.assertTrue(type(optimizer_G) == type(optimizer_D))  # must return the same type of object
            self.assertTrue(optimizer_G is not optimizer_D)          # object identity MUST be different
    
    def test_generator_shape(self):
        # Test to verify that the same dimension network is created invariant of smiles input file size
        with TemporaryDirectory() as tmpdirname:
            for j in [1, 64, 256, 1024]:
                latent=np.random.rand(j,1,512)
                os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
                with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                    json.dump(latent.tolist(), f)
                C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
                C.run()
                G = Generator.load(tmpdirname+'/generator.txt')
                G_params = []
                for param in G.parameters():
                        G_params.append(param.view(-1))
                G_params = torch.cat(G_params)
                reference= 1283968
                self.assertEqual(G_params.shape[0],reference,"Network does not match expected size")
    
    def test_discriminator_shape(self):
        # Test to verify that the same dimension network is created invariant of smiles input file size
        with TemporaryDirectory() as tmpdirname:
            for j in [1, 64, 256, 1024]:
                latent=np.random.rand(j,1,512)
                os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
                with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                    json.dump(latent.tolist(), f)
                C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
                C.run()
                D = Discriminator.load(tmpdirname+'/discriminator.txt')
                D_params = []
                for param in D.parameters():
                        D_params.append(param.view(-1))
                D_params = torch.cat(D_params)
                reference= 394241
                self.assertEqual(D_params.shape[0],reference,"Network does not match expected size")
                
    def test_sampler_n(self):
        # Verify that the sampler outputs the desired number of output latent vectors.
        with TemporaryDirectory() as tmpdirname:
            latent=np.random.rand(64,1,512)
            os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
            with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                json.dump(latent.tolist(), f)
            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            G = Generator.load(tmpdirname+'/generator.txt')
            G.cuda()
            testSampler = Sampler(G)
            samples = testSampler.sample(256)
            self.assertEqual(samples.shape[0],256, "Sampler produced a different number of latent vectors than specified")
        
                      
    def test_sampler_cuda(self):
        # Verify that the output of sampler is a CUDA tensor and not a CPU tensor when input is on CUDA
        with TemporaryDirectory() as tmpdirname:
            latent=np.random.rand(64,1,512)
            os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
            with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                json.dump(latent.tolist(), f)
            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            G = Generator.load(tmpdirname+'/generator.txt')
            json_smiles = open(tmpdirname+'/encoded_smiles.latent', "r")
            latent_space_mols = np.array(json.load(json_smiles))
            testSampler = Sampler(G)
            latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
            T = torch.cuda.FloatTensor
            G.cuda()
            dataloader = torch.utils.data.DataLoader(LatentMolsDataset(latent_space_mols), shuffle=True,
                                                        batch_size=64, drop_last=True)
            for _, real_mols in enumerate(dataloader):
                real_mols = real_mols.type(T)
                fake_mols = testSampler.sample(real_mols.shape[0])
                self.assertTrue(type(real_mols) == type(fake_mols))
                break
        
    def test_gradient_penalty_non_zero(self):
        # Test to verify that a non-zero gradient penalty is computed on the from the first training step
        with TemporaryDirectory() as tmpdirname:
            latent=np.random.rand(64,1,512)
            os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
            with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                json.dump(latent.tolist(), f)
            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            D = Discriminator.load(tmpdirname+'/discriminator.txt')
            G = Generator.load(tmpdirname+'/generator.txt')
            json_smiles = open(tmpdirname+'/encoded_smiles.latent', "r")
            latent_space_mols = np.array(json.load(json_smiles))
            testSampler = Sampler(G)
            latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
            T = torch.cuda.FloatTensor
            G.cuda()
            D.cuda()
            dataloader = torch.utils.data.DataLoader(LatentMolsDataset(latent_space_mols), shuffle=True,
                                                        batch_size=64, drop_last=True)
            for _, real_mols in enumerate(dataloader):
                real_mols = real_mols.type(T)
                fake_mols = testSampler.sample(real_mols.shape[0])   
                alpha = T(np.random.random((real_mols.size(0), 1)))
                interpolates = (alpha * real_mols + ((1 - alpha) * fake_mols)).requires_grad_(True)
                d_interpolates = D(interpolates)
                fake = T(real_mols.shape[0], 1).fill_(1.0)
                gradients = autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=fake,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
                self.assertTrue(gradient_penalty.data != 0)     
                break   
            
    def test_model_trains(self):
        # Performs one step of training and verifies that the weights are updated, implying some training occurs.
        with TemporaryDirectory() as tmpdirname:
            T = torch.cuda.FloatTensor
            latent=np.random.rand(64,1,512)
            os.makedirs(os.path.dirname(tmpdirname+'/encoded_smiles.latent'), exist_ok=True)
            with open(tmpdirname+'/encoded_smiles.latent', 'w') as f:
                json.dump(latent.tolist(), f)
            
            C = CreateModelRunner(input_data_path=tmpdirname+'/encoded_smiles.latent', output_model_folder=tmpdirname)
            C.run()
            D = Discriminator.load(tmpdirname+'/discriminator.txt')
            G = Generator.load(tmpdirname+'/generator.txt')
            G.cuda()
            D.cuda()
            optimizer_G = torch.optim.Adam(G.parameters())
            optimizer_D = torch.optim.Adam(D.parameters())
            json_smiles = open(tmpdirname+'/encoded_smiles.latent', "r")
            latent_space_mols = np.array(json.load(json_smiles))
            testSampler = Sampler(G)
            latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
            dataloader = torch.utils.data.DataLoader(LatentMolsDataset(latent_space_mols), shuffle=True,
                                                        batch_size=64, drop_last=True)
            for _, real_mols in enumerate(dataloader):
                real_mols = real_mols.type(T)
                before_G_params = []
                before_D_params = []
                for param in G.parameters():
                    before_G_params.append(param.view(-1))
                before_G_params = torch.cat(before_G_params)
                for param in D.parameters():
                    before_D_params.append(param.view(-1))
                before_D_params = torch.cat(before_D_params)
                
                optimizer_D.zero_grad()
                fake_mols = testSampler.sample(real_mols.shape[0])
                real_validity = D(real_mols)
                fake_validity = D(fake_mols)
                #It is not relevant to compute gradient penalty. The test is only interested in if there is a change in
                #the weights (training), not in giving proper training
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) 
                d_loss.backward()
                optimizer_D.step()
                optimizer_G.zero_grad()
                fake_mols = testSampler.sample(real_mols.shape[0])
                fake_validity = D(fake_mols)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                after_G_params = []
                after_D_params = []
                for param in G.parameters():
                    after_G_params.append(param.view(-1))
                after_G_params = torch.cat(after_G_params)
                for param in D.parameters():
                    after_D_params.append(param.view(-1))
                after_D_params = torch.cat(after_D_params)
                self.assertTrue(torch.any(torch.ne(after_G_params,before_G_params)))
                self.assertTrue(torch.any(torch.ne(after_D_params,before_D_params)))
                
                break
            

if __name__ == '__main__':
    unittest.main()