import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm 

def contrastive_loss(image_features, text_features, temperature=0.5):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    logits = image_features @ text_features.T / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)

def test_for_error(df):
    for row in tqdm(df.itertuples(), desc='checking for errors'):
        path = row.path
        errors = []
        try:
            a, sr = torchaudio.load(path, channels_first = True)
        except:
            errors.append(path)
    
    if len(errors) > 0:
        return errors
    else:
        return 'no errors'
    
def test_process_dl(dataloader, num_epochs=1):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Create a tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=True, position=0)
        
        for batch in progress_bar:
            # Unpack your batch here. For example:
            a, b, c = batch
            
            # Your processing code here
            # For example:
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # Update the progress bar description if needed
            # progress_bar.set_postfix(loss=loss.item())
            
            # Optional: Break the loop early for demonstration
            # if progress_bar.n >= 100:
            #     break

        progress_bar.close()

