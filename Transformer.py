import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Model from: STTRE: A Spatio-Temporal Transformer with Relative Embeddings for multivariate time series forecasting (https://github.com/AzadDeihim/STTRE)

class Transformer(nn.Module):
    def __init__(self, input_shape, output_size,
                 embed_size, num_layers, forward_expansion, heads, dropout):

        super(Transformer, self).__init__()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_elements = self.seq_len*self.num_var
        self.embed_size = embed_size
        self.element_embedding = nn.Linear(self.seq_len, embed_size*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)
        self.dropout = dropout


        self.temporal = Encoder(seq_len=self.seq_len,
                                embed_size=embed_size,
                                num_layers=num_layers,
                                heads=self.num_var,
                                device=self.device,
                                forward_expansion=forward_expansion,
                                module='temporal',
                                rel_emb=True)

        self.spatial = Encoder(seq_len=self.num_var,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=self.seq_len,
                               device=self.device,
                               forward_expansion=forward_expansion,
                               module = 'spatial',
                               rel_emb=True)

        self.spatiotemporal = Encoder(seq_len=self.seq_len*self.num_var,
                                      embed_size=embed_size,
                                      num_layers=num_layers,
                                      heads=heads,
                                      device=self.device,
                                      forward_expansion=forward_expansion,
                                      module = 'spatiotemporal',
                                      rel_emb=True)


        # consolidate embedding dimension
        self.fc_out1 = nn.Linear(embed_size, embed_size//2)
        self.fc_out2 = nn.Linear(embed_size//2, 1)

        #prediction
        self.out = nn.Linear((self.num_elements*3), output_size)


    def forward(self, x):
        batch_size = len(x)

        #process/embed input for temporal module
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, self.dropout)

        #process/embed input for spatial module
        x_spatial = torch.transpose(x, 1, 2).reshape(batch_size, self.num_var, self.seq_len)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.element_embedding(x_spatial).reshape(batch_size, self.num_elements, self.embed_size)
        x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, self.dropout)


        #process/embed input for spatio-temporal module
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var* self.seq_len).to(self.device)
        x_spatio_temporal = self.element_embedding(x).reshape(batch_size, self.seq_len* self.num_var, self.embed_size)
        x_spatio_temporal = F.dropout(self.pos_embedding(positions) + x_spatio_temporal, self.dropout)



        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out3 = self.spatiotemporal(x_spatio_temporal)
        out = torch.cat((out1, out2, out3), 1)
        out = self.fc_out1(out)
        out = F.leaky_relu(out)
        out = self.fc_out2(out)
        out = F.leaky_relu(out)
        out = torch.flatten(out, 1)
        out = self.out(out)

        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)

        if module == 'spatial' or module == 'temporal':
            self.norm1 = nn.BatchNorm1d(seq_len*heads)
            self.norm2 = nn.BatchNorm1d(seq_len*heads)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )


    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)

        return out
    

class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads, seq_len, module, rel_emb):
        super(SelfAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (modules.__contains__(module)), "Invalid module"


        if module == 'spatial' or module == 'temporal':
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size], device=self.device))


        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim], device=self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, device=self.device)

    def forward(self, x):
        N, _, _ = x.shape

        #non-shared weights between heads for spatial and temporal modules
        if self.module == 'spatial' or self.module == 'temporal':
            values = self.values(x)
            keys = self.keys(x)
            queries = self.queries(x)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)

        #shared weights between heads for spatio-temporal module
        else:
            values, keys, queries = x, x, x
            values = values.reshape(N, self.seq_len, self.heads, self.head_dim)
            keys = keys.reshape(N, self.seq_len, self.heads, self.head_dim)
            queries = queries.reshape(N, self.seq_len, self.heads, self.head_dim)
            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)


        if self.rel_emb:
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1,2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = self.custom_triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S

        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            mask = self.custom_triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device),
                    1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        #attention(N x Heads x Q_Len x K_len)
        #values(N x V_len x Heads x Head_dim)
        #z(N x Q_len x Heads*Head_dim)

        if self.module == 'spatial' or self.module == 'temporal':
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len*self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len, self.heads*self.head_dim)

        z = self.fc_out(z)

        return z


    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = self.custom_triu(torch.ones(L, L, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]
    
    
    def custom_triu(self, input, diagonal=0):
        """
        Returns the upper triangular part of an N-dimensional tensor.
        
        Args:
            input (Tensor): The input tensor.
            diagonal (int, optional): The diagonal above which to consider values.
                The main diagonal is 0, diagonals above are positive, and those below are negative.
        
        Returns:
            Tensor: A new tensor containing the upper triangular part of the input tensor.
        """
        return torch.triu(input, diagonal)
        #Ensure the input is a tensor
        if not torch.is_tensor(input):
            raise TypeError("input must be a tensor")
        
        # Create a tensor of the same shape as input, filled with zeros
        result = torch.zeros_like(input)
        
        # Get the dimensions of the input tensor
        *leading_dims, rows, cols = input.shape
        
        # Number of leading elements (total elements in the leading dimensions)
        num_leading_elements = input.numel() // (rows * cols)
        
        # Iterate over the leading elements
        for index in range(num_leading_elements):
            # Convert the linear index to a multi-dimensional index for leading dimensions
            idx = np.unravel_index(index, leading_dims)
            for i in range(rows):
                for j in range(cols):
                    if j >= i + diagonal:
                        result[idx + (i, j)] = input[idx + (i, j)]
        
        return result
    


class TriuFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, diagonal):
        return g.op("TriuFunc", x, diagonal_i=diagonal)

    @staticmethod
    def forward(ctx, x, diagonal=0):
        mask = torch.triu(torch.ones_like(x), diagonal)
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask
    

class Encoder(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device,
                 forward_expansion, module, output_size=1,
                 rel_emb=True):
        super(Encoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [
             TransformerBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb = rel_emb)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)

        out = self.fc_out(out)
        return out