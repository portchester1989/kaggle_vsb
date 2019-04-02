import torch
from torch.utils.data import Dataset 

class AugmentedData(Dataset):
    def __init__(self,signal_data,meta_data):
        #self.meta_data = pd.read_csv("../ken_hayashima1989/metadata_train.csv")
        self.positive_indices = meta_data[meta_data.target == 1].index - 5000
        self.negative_indices = meta_data[meta_data.target == 0].index - 5000
        self.positive_sample = signal_data[:,self.positive_indices]
        self.negative_sample = signal_data[:,self.negative_indices]
        self.X,self.y_a,self.y_b,self.lam = self.mix_up()
        #self.X = results[1:,:]
        #self.Y = results[0]
    def mix_up(self,slice_num = 30,alpha = .2,iteration_num = 15):
        mixed_results =[]
        for i in range(iteration_num):
            positive = np.vstack([np.ones((1,self.positive_sample.shape[1])),self.positive_sample])
            negative = np.vstack([np.zeros((1,self.negative_sample.shape[1])),self.negative_sample]) 
            np.random.shuffle(positive.T)
            np.random.shuffle(negative.T)
            signals_to_be_mixed = np.hstack([positive[:,:slice_num],negative[:,:slice_num]])
            signal_A = np.random.permutation(signals_to_be_mixed.T)
            signal_B = np.random.permutation(signals_to_be_mixed.T)
            mixing_coef = np.random.beta(alpha,alpha,slice_num * 2)
            #mixed_results.append(signal_A.T * mixing_coef + signal_B.T * (1 - mixing_coef))
            if i == 0:
                mixed_result = signal_A.T[1:,:] * mixing_coef + signal_B.T[1:,:] * (1 - mixing_coef)
                y_a = signal_A.T[0]
                y_b = signal_B.T[0]
                lam = mixing_coef
            else:
                mixed_result = np.hstack([mixed_result,signal_A.T[1:,:] * mixing_coef + signal_B.T[1:,:] * (1 - mixing_coef)])
                y_a = np.concatenate([y_a,signal_A.T[0]])
                y_b = np.concatenate([y_b,signal_B.T[0]])
                lam = np.concatenate([lam,mixing_coef])
        return mixed_result,y_a,y_b,lam
    def __len__(self):
        return len(self.lam)
    def __getitem__(self,idx):
        X = self.X[:,idx].T
        X = np.expand_dims(X,axis = 1)
        Y_A = self.y_a[idx]
        Y_B = self.y_b[idx]
        lam = self.lam[idx]
        return {'X':X,'Y_A':Y_A,'Y_B':Y_B,'lam':lam}
