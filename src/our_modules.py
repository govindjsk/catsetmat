import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence."""
    # Expand to fit the shape of key query attention matrix.
    pm_q = seq_q.eq(0)
    pm_k = seq_k.eq(0)
    pm_q_ = pm_q.unsqueeze(1).expand(-1, seq_k.shape[1], -1)
    pm_k_ = pm_k.unsqueeze(1).expand(-1, seq_q.shape[1], -1)
    padding_mask = pm_q_.transpose(1, 2) | pm_k_
    return padding_mask


class PositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            dims,
            dropout=None,
            reshape=False,
            use_bias=True,
            residual=False,
            layer_norm=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_stack = []
        self.dims = dims
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, use_bias))
            self.add_module("PWF_Conv%d" % (i), self.w_stack[-1])
        self.reshape = reshape
        self.layer_norm = nn.LayerNorm(dims[-1])

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.residual = residual
        self.layer_norm_flag = layer_norm

    def forward(self, x):
        output = x.transpose(1, 2)

        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)

        output = self.w_stack[-1](output)
        output = output.transpose(1, 2)

        if self.reshape:
            output = output.view(output.shape[0], -1, 1)

        if self.dims[0] == self.dims[-1]:
            # residual
            if self.residual:
                output += x

            if self.layer_norm_flag:
                output = self.layer_norm(output)

        return output


class FeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, dims, dropout=True, reshape=False, use_bias=True):
        super(FeedForward, self).__init__()
        self.w_stack = []
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
            self.add_module("FF_Linear%d" % (i), self.w_stack[-1])

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reshape = reshape

    def forward(self, x):
        output = x
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)

        if self.reshape:
            output = output.view(output.shape[0], -1, 1)

        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def masked_softmax(self, vector: torch.Tensor,
                       mask: torch.Tensor,
                       dim: int = -1,
                       memory_efficient: bool = False,
                       mask_fill_value: float = -1e32) -> torch.Tensor:
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            # pdb.set_trace()
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside
                # the mask, we zero these out.
                masked_vector = vector.masked_fill(mask.bool(), -float('inf'))
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
                result = result * (1 - mask).bool()
                # result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
                result = result * (1 - mask).bool()
        return result

    def forward(self, q, k, v, diag_mask, mask=None):
        # pdb.set_trace()
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        # if mask is not None:
        #     attn = attn.masked_fill(mask, -float('inf'))
        attn = self.masked_softmax(attn, mask, dim=-1, memory_efficient=True)
        # attn = torch.nn.functional.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout, diag_mask, input_dim, static_flag=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.static_flag = static_flag

        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
        if self.static_flag:
            self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.diag_mask_flag = diag_mask
        self.diag_mask = None

    def pass_(self, inputs):
        return inputs

    def forward(self, q, k, v, diag_mask, mask=None):
        # pdb.set_trace()
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual_dynamic = q
        residual_static = v
        q = self.layer_norm1(q)
        k = self.layer_norm2(k)
        v = self.layer_norm3(v)

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        n = sz_b * n_head
        """change masking matrix from len_v to len_q for cross attentions"""
        self.diag_mask = (torch.ones((len_q, len_v), device=device))
        if self.diag_mask_flag == 'True':
            self.diag_mask -= torch.eye(len_q, len_v, device=device)
        self.diag_mask = self.diag_mask.repeat(n, 1, 1)
        diag_mask = self.diag_mask
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
        dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
        dynamic = dynamic.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
        if self.static_flag:

            static = v.view(n_head, sz_b, len_k, d_v)
            static = static.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1)  # b x lq x (n*dv)
            static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
            return dynamic, static, attn
        else:

            return dynamic, attn


class CrossAttention(nn.Module):
    """A self-attention layer + 2 layered pff"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.slf_attn_lv1_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)
        self.slf_attn_lv1_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)

        self.cross_attn_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.cross_attn_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.slf_attn_lv2_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.slf_attn_lv2_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.pff_U1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)

        self.pff_U2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

        self.pff_V1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)

        self.pff_V2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

    # self.dropout = nn.Dropout(0.2)

    def forward(self, dynamic_1, dynamic_2, static_1, static_2, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1,
                slf_attn_mask2, non_pad_mask1, non_pad_mask2):
        """here the static_1 refer to the input embeddings of U_side while static_2 relates the embeddings of V sides
        and dynamic_1 refer to query embedding of u side(input) and  dynamic_2 refers to embeddings of V sides """

        """only change is now self attention mask and non pad_mask"""  ########
        # pdb.set_trace()
        dynamic1u, static1, attn_lv1u = self.slf_attn_lv1_u(dynamic_1, static_1, static_1, diag_mask=None,
                                                            mask=slf_attn_mask1)
        dynamic1v, static2, attn_lv1v = self.slf_attn_lv1_v(dynamic_2, static_2, static_2, diag_mask=None,
                                                            mask=slf_attn_mask2)
        dynamic2u, cr_attn_u = self.cross_attn_u(dynamic1u, dynamic1v, dynamic1v, diag_mask=None, mask=crs_attn_mask1)
        dynamic2v, cr_attn_v = self.cross_attn_v(dynamic1v, dynamic1u, dynamic1u, diag_mask=None, mask=crs_attn_mask2)
        dynamic3u, attn_lv2u = self.slf_attn_lv2_u(dynamic2u, dynamic2u, dynamic2u, diag_mask=None, mask=slf_attn_mask1)
        dynamic3v, attn_lv2v = self.slf_attn_lv2_v(dynamic2v, dynamic2v, dynamic2v, diag_mask=None, mask=slf_attn_mask2)
        output_attn = [attn_lv1u, attn_lv1v, attn_lv2u, attn_lv2v, cr_attn_u, cr_attn_v]

        # dynamic1, cr_attn1 = self.mul_head_attn_forward(dynamic_1, static_2, static_2, diag_mask=None,
        #                                                 mask=crs_attn_mask1)
        # dynamic2, cr_attn2 = self.mul_head_attn_backward(dynamic_2, static_1, static_1, diag_mask=None,
        #                                                  mask=crs_attn_mask2)
        # static1, slf_attn1 = self.mul_head_attn_selfU(dynamic_1, static_1, static_1, diag_mask=None,
        #                                               mask=slf_attn_mask1)
        # static2, slf_attn2 = self.mul_head_attn_selfV(dynamic_2, static_2, static_2, diag_mask=None,
        #                                               mask=slf_attn_mask2)

        dynamic1 = self.pff_U1(dynamic3u * non_pad_mask1) * non_pad_mask1
        static1 = self.pff_U2(static1 * non_pad_mask1) * non_pad_mask1
        dynamic2 = self.pff_V1(dynamic3v * non_pad_mask2) * non_pad_mask2
        static2 = self.pff_V2(static2 * non_pad_mask2) * non_pad_mask2
        return dynamic1, static1, dynamic2, static2, output_attn


class CrossAttentionSimple(nn.Module):
    """A self-attention layer + 2 layered pff"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.slf_attn_lv1_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)
        self.slf_attn_lv1_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)

        self.cross_attn_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)
        self.cross_attn_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        # self.slf_attn_lv2_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
        #                                          diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)
        #
        # self.slf_attn_lv2_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
        #                                          diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.pff_U1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_U2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)
        self.pff_V1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_V2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

    # self.dropout = nn.Dropout(0.2)

    def forward(self, dynamic_1, dynamic_2, static_1, static_2, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1,
                slf_attn_mask2, non_pad_mask1, non_pad_mask2):
        """here the static_1 refer to the input embeddings of U_side while static_2 relates the embeddings of V sides
        and dynamic_1 refer to query embedding of u side(input) and  dynamic_2 refers to embeddings of V sides """

        """only change is now self attention mask and non pad_mask"""  ########
        # pdb.set_trace()
        dynamic1u, static1, attn_lv1u = self.slf_attn_lv1_u(dynamic_1, static_1, static_1, diag_mask=None,
                                                            mask=slf_attn_mask1)

        dynamic1v, static2, attn_lv1v = self.slf_attn_lv1_v(dynamic_2, static_2, static_2, diag_mask=None,
                                                            mask=slf_attn_mask2)

        dynamic2u, cr_attn_u = self.cross_attn_u(dynamic1u, dynamic1v, dynamic1v, diag_mask=None, mask=crs_attn_mask1)

        dynamic2v, cr_attn_v = self.cross_attn_v(dynamic1v, dynamic1u, dynamic1u, diag_mask=None, mask=crs_attn_mask2)

        # dynamic3u, attn_lv2u = self.slf_attn_lv2_u(dynamic2u, dynamic2u, dynamic2u, diag_mask=None,
        #                                            mask=slf_attn_mask1)
        # dynamic3v, attn_lv2v = self.slf_attn_lv2_v(dynamic2v, dynamic2v, dynamic2v, diag_mask=None,
        #                                            mask=slf_attn_mask2)

        output_attn = [attn_lv1u, attn_lv1v, cr_attn_u, cr_attn_v]

        # dynamic1, cr_attn1 = self.mul_head_attn_forward(dynamic_1, static_2, static_2, diag_mask=None,
        #                                                 mask=crs_attn_mask1)
        # dynamic2, cr_attn2 = self.mul_head_attn_backward(dynamic_2, static_1, static_1, diag_mask=None,
        #                                                  mask=crs_attn_mask2)
        # static1, slf_attn1 = self.mul_head_attn_selfU(dynamic_1, static_1, static_1, diag_mask=None,
        #                                               mask=slf_attn_mask1)
        # static2, slf_attn2 = self.mul_head_attn_selfV(dynamic_2, static_2, static_2, diag_mask=None,
        #                                               mask=slf_attn_mask2)

        dynamic1 = self.pff_U1(dynamic2u * non_pad_mask1) * non_pad_mask1
        static1 = self.pff_U2(static1 * non_pad_mask1) * non_pad_mask1
        dynamic2 = self.pff_V1(dynamic2v * non_pad_mask2) * non_pad_mask2
        static2 = self.pff_V2(static2 * non_pad_mask2) * non_pad_mask2
        return dynamic1, static1, dynamic2, static2, output_attn


class Classifier(nn.Module):
    """a classifier is the main model for embeddings"""

    def __init__(self, n_head, d_model, d_k, d_v, node_embedding1, node_embedding2, diag_mask, bottle_neck, **args):
        super().__init__()

        self.pff_classifier1 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier2 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier3 = PositionwiseFeedForward([1, 1], reshape=True, use_bias=True)
        # self.pff_classifier3 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)

        """remove positional embedding"""  ###########

        self.node_embedding1 = node_embedding1
        self.node_embedding2 = node_embedding2
        self.encode1 = CrossAttention(n_head, d_model, d_k, d_v, dropout_mul=0.4, dropout_pff=0.4,
                                      diag_mask=diag_mask, bottle_neck=bottle_neck)
        # self.encode1 = CrossAttentionSimple(n_head, d_model, d_k, d_v, dropout_mul=0.4, dropout_pff=0.4,
        #                                     diag_mask=diag_mask, bottle_neck=bottle_neck)

        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)

    def get_node_embeddings(self, x, mode, return_recon=False):

        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        # print(torch.max(x), torch.min(x))
        if mode == 1:
            x, recon_loss = self.node_embedding1(x.view(-1))
        else:
            x, recon_loss = self.node_embedding2(x.view(-1))
        if return_recon:
            return x.view(sz_b, len_seq, -1), recon_loss
        else:
            return x.view(sz_b, len_seq, -1)

    def get_embedding(self, x, y, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1, slf_attn_mask2, non_pad_mask1,
                      non_pad_mask2, return_recon=False):
        if return_recon:
            x, recon_loss1 = self.get_node_embeddings(x, 1, return_recon)
            y, recon_loss2 = self.get_node_embeddings(y, 2, return_recon)
        else:
            x = self.get_node_embeddings(x, 1, return_recon)
            y = self.get_node_embeddings(y, 2, return_recon)
            recon_loss1, recon_loss2 = None, None
        dynamic1, static1, dynamic2, static2, output_attn = self.encode1(x, y, x, y, crs_attn_mask1, crs_attn_mask2,
                                                                         slf_attn_mask1, slf_attn_mask2, non_pad_mask1,
                                                                         non_pad_mask2)
        if return_recon:
            return dynamic1, static1, dynamic2, static2, output_attn, recon_loss1, recon_loss2
        else:
            return dynamic1, static1, dynamic2, static2, output_attn

    def forward(self, x, y, mask=None, get_outlier=None, return_recon=False):
        x = x.long()
        # pdb.set_trace()
        cr_attn_mask1 = get_attn_key_pad_mask(seq_k=y, seq_q=x)
        slf_attn_mask1 = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask1 = get_non_pad_mask(x)
        cr_attn_mask2 = get_attn_key_pad_mask(seq_k=x, seq_q=y)
        slf_attn_mask2 = get_attn_key_pad_mask(seq_k=y, seq_q=y)
        non_pad_mask2 = get_non_pad_mask(y)
        if return_recon:
            dynamic1, static1, dynamic2, static2, \
                recon_loss1, recon_loss2 = self.get_embedding(x, y, cr_attn_mask1, cr_attn_mask2,
                                                              slf_attn_mask1, slf_attn_mask2,
                                                              non_pad_mask1, non_pad_mask2, return_recon)
        else:
            dynamic1, static1, dynamic2, static2, \
                output_attn = self.get_embedding(x, y, cr_attn_mask1, cr_attn_mask2, slf_attn_mask1, slf_attn_mask2,
                                                 non_pad_mask1, non_pad_mask2, return_recon)

        dynamic1 = self.layer_norm1(dynamic1)
        static1 = self.layer_norm2(static1)
        dynamic2 = self.layer_norm3(dynamic2)
        static2 = self.layer_norm4(static2)
        # sz_b, len_seq, dim = dynamic1.shape
        # pdb.set_trace()
        # output=torch.cat([((dynamic1-static1)**2),((dynamic2-static2)**2)],dim=1)
        output1 = self.pff_classifier1((dynamic1 - static1) ** 2)
        output2 = self.pff_classifier2((dynamic2 - static2) ** 2)
        # output = dynamic1**2+dynamic2**2
        # output1 = self.pff_classifier(dynamic1**2)
        # output2 = self.pff_classifier(dynamic2**2)
        output = torch.cat([output1, output2], axis=1)
        output = self.pff_classifier3(output)
        # pdb.set_trace()
        # output = torch.sigmoid(torch.cat([output1,output2],axis=1))
        output = torch.sigmoid(output)

        # embedding_after_attn = [[static1,dynamic1],[static2,dynamic2], output_attn]
        embedding_after_attn = None
        non_pad_mask = torch.cat([non_pad_mask1, non_pad_mask2], axis=1)
        if get_outlier is not None:
            k = get_outlier
            outlier = ((1 - output) * non_pad_mask).topk(k, dim=1, largest=True, sorted=True)[1]
            return outlier.view(-1, k)

        mode = 'first'
        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output

        elif mode == 'sum':
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output /= mask_sum

        elif mode == 'first':
            output = output[:, 0, :]

        if return_recon:
            return output, None, embedding_after_attn
        else:
            return output, embedding_after_attn

    def return_embeddings(self, x, mode):
        # x must be tensor of elements (index)
        if mode == 1:
            return self.node_embedding1[x]
        else:
            return self.node_embedding2[x]

    def save_trained_embeddings(self, file_path):
        file = {"first_set_graph": self.node_embedding1, "second_set_graph": self.node_embedding2}
        torch.save(file, file_path)
