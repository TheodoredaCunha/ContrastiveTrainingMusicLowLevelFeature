import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from dataset import NormalizeToMinusOneToOne


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ContrastiveDataet(Dataset):
    def __init__(self, dataframe, model_path = 'vq_vae/model4.pth', bert_model_name = 'bert-base-cased', tokenizer_max_length = 128):
        self.dataframe = dataframe
        self.saved_weights = torch.load(model_path)['model']
        
        self.model_name = bert_model_name
        #self.bert_tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.max_length = tokenizer_max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.dataframe.iloc[idx, 0]  # Assumes the path is in the first column
        timeseries = self.to_melspec(path)
        text = self.dataframe.iloc[idx, 1]  # Assumes the caption is in the second column
        input_ids, attention_masks = self.text_embedding(text)
        return timeseries, input_ids, attention_masks


    '''Text Embedding Methods'''
    def text_embedding(self, text):
        inputs = self.bert_tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask

    '''Music Embedding Methods'''
    def to_melspec(self, path):
        full_waveform, sr = torchaudio.load(path, channels_first=True)
        full_waveform = self.pad_or_trim(full_waveform,1322000)

        # cutting the 30 second clip to 5 second clips (6 pieces)
        waveforms_list = torch.chunk(full_waveform, 6, 1)

        res = []
        for i, waveform in enumerate(waveforms_list):
            melspec = self.transform(waveform, sr)
            melspec = torch.mean(melspec, dim=0)
            dim1, dim2 = melspec.size()
            melspec = melspec.view(-1, 1, dim1, dim2).to(device)
            # z = self.model.encode(melspec)
            # z_q, *_ = self.model.vq(z)
            res.append(melspec)
        res = torch.cat(res, dim = 0)

            
        return res

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
        melspec = MelSpectrogram(n_fft = 2048)(waveform)
        melspec = AmplitudeToDB()(melspec)
        melspec = NormalizeToMinusOneToOne()(melspec)

        return melspec
