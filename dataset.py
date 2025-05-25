import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Spectrogram


class NormalizeToMinusOneToOne:
    def __call__(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max > tensor_min:
            normalized_tensor = 2 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1
        else:
            normalized_tensor = tensor - tensor_min
        
        return normalized_tensor
    
class AudioDataset(Dataset):
    def __init__(self, filenames, channels_first = True, nfft = 2048, return_time_series = True, normalize = True, return_filename = False):

        self.file_list = filenames
        self.channels_first = channels_first
        self.n_fft = nfft
        self.return_time_series = return_time_series
        self.normalize = normalize
        self.return_filename = return_filename

        if self.channels_first:
            self.chunk_dim = 1
        else:
            self.chunk_dim = 0

        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # When using the FMA Dataset, 30 second MP3 files will be used

        full_waveform, sample_rate = torchaudio.load(self.file_list[idx], channels_first=True)
        full_waveform = self.pad_or_trim(full_waveform,1322000)

        # cutting the 30 second clip to 5 second clips (6 pieces)
        waveforms_list = torch.chunk(full_waveform, 6, self.chunk_dim)
        res = []

        
        if self.return_time_series:
            for i, waveform in enumerate(waveforms_list):
                melspec = self.transform(waveform, sample_rate)
                melspec = torch.mean(melspec, dim=0)

                res.append({
                    'melspec': melspec,
                    'sr': sample_rate
                })
                
            if self.return_filename:
                return melspec, self.file_list[idx]

            return res
        else:
            melspec = self.transform(waveforms_list[0], sample_rate)
            melspec = torch.mean(melspec, dim = 0)


            if self.return_filename:
                return melspec, self.file_list[idx]
            
            return melspec




    def pad_or_trim(self, tensor, target_length):
        current_length = tensor.shape[1]
        
        if current_length > target_length:
            # Trim the tensor
            return tensor[:, :target_length]
        elif current_length < target_length:
            # Pad the tensor
            padding = target_length - current_length
            pad_tensor = torch.nn.functional.pad(tensor, (0, padding), mode='constant', value=0)
            return pad_tensor
        else:
            # No need to pad or trim
            return tensor
    
    def transform(self, waveform, sr):
        # Convert Audio to Melspectrograms
        melspec = MelSpectrogram(n_fft = self.n_fft)(waveform)
        melspec = AmplitudeToDB()(melspec)

        if self.normalize:
            melspec = NormalizeToMinusOneToOne()(melspec)

        return melspec

