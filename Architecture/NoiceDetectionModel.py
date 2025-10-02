from torch import nn 
import torch 
from torch.nn import functional as F
from torchvision import transforms as T


class PatchEmbedding (nn.Module) :
  def __init__ (self,image_size,patch_size,embedding_size) :
    super().__init__()
    self.projection_layers = nn.Conv2d(in_channels=3,out_channels=embedding_size,kernel_size=patch_size,stride=patch_size)
    self.n_patch = (image_size // patch_size)**2

  def forward(self,x) :
    x = self.projection_layers(x)
    x = x.flatten(2)
    x = x.transpose(1,2)
    return x

class PositionalEmbedding (nn.Module) :
  def __init__ (self,n_patch,embedding_size) :
    super().__init__()
    self.n_patch = n_patch
    self.position = nn.Parameter(torch.normal(0.0,0.02,size=(1,self.n_patch + 1,embedding_size)))
    self.cls_token = nn.Parameter(torch.normal(0.0,0.02,size=(1,1,embedding_size)))
    self.embedding_size = embedding_size

  def forward(self,x) :
    batch = x.shape[0]
    cls_token = torch.broadcast_to(self.cls_token,(batch,1,self.embedding_size))
    x = torch.cat((cls_token,x),dim=1)
    x = x + self.position

    return x

class BlockTransformers (nn.Module) :
  def __init__ (self,d_model,num_head,ffn_dim,droprate= 0.1) :
    super().__init__()
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.MHA = nn.MultiheadAttention(embed_dim=d_model,num_heads=num_head,dropout=droprate)
    self.FeedFordward = nn.Sequential(
        nn.Linear(d_model,ffn_dim),
        nn.GELU(),
        nn.Linear(ffn_dim,d_model)
    )
    self.drop_out = nn.Dropout(droprate)

  def forward(self,x) :
    attn = self.norm1(x)
    attn,_ = self.MHA(attn,attn,attn)
    x = x+attn

    ffn = self.norm2(x)
    ffn = self.FeedFordward(x)
    ffn = self.drop_out(x)
    x = x+ffn
    return x
  
class NoiceDetectorModel (nn.Module) :
  def __init__(self,image_size,d_model,num_head,ffn_dim,droprate= 0.1) :
    super().__init__()
    self.patch_embedding = PatchEmbedding(image_size=image_size,patch_size=16,embedding_size=d_model)
    self.positional_embedding = PositionalEmbedding(self.patch_embedding.n_patch,d_model)
    self.blocklayers = nn.Sequential(
        BlockTransformers(d_model,num_head,ffn_dim,droprate),
        BlockTransformers(d_model,num_head,ffn_dim,droprate))
    self.linear1 = nn.Linear(d_model,128)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(128,3)
  def forward(self,x) :
    x = self.patch_embedding(x)
    x = self.positional_embedding(x)
    x = self.blocklayers(x)
    x = x[:,-1,:]
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    return x

class ModelRunners :
  def __init__(self,path) :
    self.Model = NoiceDetectorModel(image_size=384,d_model=256,num_head=4,ffn_dim=784)
    self.__checkpoint = torch.load(path)
    self.Model.load_state_dict(self.__checkpoint)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.Model.to(self.device)
    self.Model.eval()
    self.transform =T.Compose([
          T.ToTensor(),
          T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
      ])
  
  def modelrun (self,x_target) :
    if not isinstance(x_target,torch.Tensor) :
      x_target = self.transform(x_target)
      x_target = torch.unsqueeze(x_target,dim=0)
    
    with torch.no_grad() :
      pred = self.Model(x_target)
      pred = F.softmax(pred,dim=-1)
      
    if isinstance(pred,torch.Tensor) :
      return pred.detach().numpy()
    
    else :
      return pred 
    