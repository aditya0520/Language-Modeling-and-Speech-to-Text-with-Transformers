import torch

class Softmax:

    '''
    DO NOT MODIFY! AN INSTANCE IS ALREADY SET IN THE Attention CLASS' CONSTRUCTOR. USE IT!
    Performs softmax along the last dimension
    '''
    def forward(self, Z):

        z_original_shape = Z.shape

        self.N = Z.shape[0]*Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape

        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))
        
        for i in range(self.N):

            J = torch.zeros((self.C, self.C))

            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)

class Attention:
        
        def __init__(self, weights_keys, weights_queries, weights_values):

            """
            Initialize instance variables. Refer to writeup for notation.
            input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

            Argument(s)
            -----------
            
            weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys 
            weights_queries (torch.tensor, dim = (D X D_k)): weight matrix for queries 
            weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values 
            
            """

            # Store the given weights as parameters of the class.
            self.W_k    = weights_keys
            self.W_q    = weights_queries
            self.W_v    = weights_values

            # Use this object to perform softmax related operations.
            # It performs softmax over the last dimension which is what you'll need.
            self.softmax = Softmax()
            
        def forward(self, X):

            """
            Compute outputs of the self-attention layer.
            Stores keys, queries, values, raw and normalized attention weights.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            X (torch.tensor, dim = (B, T, D)): Input batch

            Return
            ------
            X_new (torch.tensor, dim = (B, T, D_v)): Output batch

            """

            self.X = X
    
            # Step 1: Compute the Query, Key, and Value matrices
            self.Q = torch.matmul(X, self.W_q) 
            self.K = torch.matmul(X, self.W_k)
            self.V = torch.matmul(X, self.W_v) 
            
            # Step 2: Calculate unnormalized attention scores
            self.A_w    = torch.matmul(self.Q, self.K.permute(0, 2, 1)) # (B, T, T)
            
            # Step 3: Apply causal mask to prevent attending to future tokens
            upper_triangular_mask = torch.triu(torch.ones_like(self.A_w), diagonal=1)
            attn_mask = self.A_w - 1e9 * upper_triangular_mask  # Apply mask
            
            # Step 4: Normalize the attention scores with softmax, scaled by sqrt(D_k)
            self.A_sig = self.softmax.forward(attn_mask / torch.sqrt(torch.tensor(self.W_k.shape[-1], dtype=torch.float32, device=attn_mask.device)))
            
            # Step 5: Compute the attention context
            X_new = torch.matmul(self.A_sig, self.V)  # (B, T, D_v)
            
            return X_new
            
        def backward(self, dLdXnew):

            """
            Backpropogate derivatives through the self-attention layer.
            Stores derivatives wrt keys, queries, values, and weight matrices.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the divergence wrt attention layer outputs

            Return
            ------
            dLdX (torch.tensor, dim = (B, T, D)): Derivative of the divergence wrt attention layer inputs

            """

            # Derivatives wrt attention weights (raw and normalized)

            dLdA_sig       =  torch.matmul(dLdXnew, self.V.permute(0,2,1))
            dLdA_w         =  self.softmax.backward(dLdA_sig) * 1 / self.X.shape[2] ** 0.5

            dLdA_w = (1 / torch.sqrt(torch.tensor(self.W_k.shape[-1], dtype=torch.float32, device=dLdA_sig.device))) * self.softmax.backward(dLdA_sig)

            # Derivatives wrt keys, queries, and value
            
            self.dLdV      =  torch.bmm(self.A_sig.transpose(1,2), dLdXnew)
            self.dLdK      =  torch.bmm(dLdA_w.transpose(1,2), self.Q)
            self.dLdQ      =  torch.bmm(dLdA_w, self.K)

            # Dervatives wrt weight matrices
            # Remember that you need to sum the derivatives along the batch dimension.

            self.dLdWq     =  torch.sum(torch.bmm(self.X.transpose(1,2), self.dLdQ), dim=0)
            self.dLdWv     =  torch.sum(torch.bmm(self.X.transpose(1,2), self.dLdV), dim=0)
            self.dLdWk     =  torch.sum(torch.bmm(self.X.transpose(1,2), self.dLdK), dim=0)

            # Derivative wrt input

            W_v_expanded = self.W_v.T.unsqueeze(0).expand(self.dLdV.size(0), -1, -1)
            W_k_expanded = self.W_k.T.unsqueeze(0).expand(self.dLdK.size(0), -1, -1)
            W_q_expanded = self.W_q.T.unsqueeze(0).expand(self.dLdQ.size(0), -1, -1)

            # Calculate dLdX with expanded weight matrices
            dLdX = torch.bmm(self.dLdV, W_v_expanded) + \
                torch.bmm(self.dLdK, W_k_expanded) + \
                torch.bmm(self.dLdQ, W_q_expanded)
            
            return dLdX