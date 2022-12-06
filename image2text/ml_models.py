import numpy as np
import torch
from torch import nn
import torchvision
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

class DecoderWithAttention(nn.Module):
    """
    画像の特徴量とキャプションの単語から次の単語を求めるデコーダー。
    
    Attributes
    ----------
    encoder_dim : int
            エンコーダーの次元数。
    attention_dim : int
        アテンションの次元数。
    embed_dim : int
        埋め込み層の次元数。
    decoder_dim : int
        デコーダーの次元数。
    vocab_size : int
        用いる全単語の数。
    dropout 
        ドロップアウト層。
    attention
        アテンション層。
    embedding
        埋め込み層。
    decode_step
        LSTMCell層。
    init_h
        encoder_dimの次元の入力をdecoder_dimに変換する全結合層。
    init_c 
        encoder_dimの次元の入力をdecoder_dimに変換する全結合層。
    f_beta
        decoder_dimの次元の入力をencoder_dimに変換する全結合層。
    sigmoid
        活性化関数Sigmoid。
    fc
        decoder_dimの次元の入力をvocab_sizeに変換する全結合層。
    """
    def __init__(self, attention_dim : int, embed_dim : int, decoder_dim : int, vocab_size : int,
                 encoder_dim : int =1280, dropout : int =0.5):
        """
        次元数や層の初期化。
        
        Parameters
        --------------
        attention_dim : int
            アテンションの次元数。
        embed_dim : int
          埋め込み層の次元数。
        decoder_dim : int
            デコーダーの次元数。
        vocab_size : int
          用いる全単語の数。
        encoder_dim : int
            エンコーダーの次元数。
        dropout : int 
            ドロップアウトの割合。
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights() 

    def init_weights(self):
        """
          重みの初期化。
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """ 
          埋め込み層の重みを初期化する。

          Parameters
          --------------
          embeddings : TensorType["batch", "embed_dim", float]
              読み込む重み。
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune : bool =True) -> None:
        """ 
        埋め込み層をファインチューニングする。

          Parameters
          --------------
          fine_tune : bool
              Trueなら埋め込み層をファインチューニングする。Falseならしない。
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out) -> None:
        """ 
        隠れ層の初期化をする。

        Parameters
        --------------
        encoder_out : TensorType["batch", "width", "height", "encoder_dim" float]
            Trueなら埋め込み層をファインチューニングする。Falseならしない。
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions : "list", caption_lengths : "list"):
        """ 
        順伝播を行う。

        Parameters
        --------------
        encoder_out : TensorType["batch", "width", "height", "encoder_dim" float]
            Trueなら埋め込み層をファインチューニングする。Falseならしない。
        encoded_captions : list[list[int]]
            batch_size×cap_per_imgの要素数の配列の各要素にそのキャプションの各単語がエンコードされた数字として格納されている。
        caption_lengths : list[int]
            各キャプションの長さを格納する配列。

        Returns
        ---------
        predictions : TensorType["batch", "max_decode_lengths", "vocab_size", float]
            予測値のテンソル。
        encoded_captions : list[list[int]]
            ソート後のencoded_captions。
        decode_lengths : list[int]
            caption_lengthsの全要素を-1した配列。
        alphas : TensorType["batch_size", "max(decode_lengths)", "num_pixels", float]
            全てのバッチの全ての予測単語に対するalphaの値を格納する配列。
        sort_ind : list[int]
            キャプションをキャプションの長さごとに降順に並べ替えたときのindexを示す配列。
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)#(batch_size, num_of_pixels,encoder_dim)に変形
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):#max(decode_lengths)最大単語数?
            #バッチサイズは最初は最大で、次は最後を使わない。その次はその前を使わない。
            batch_size_t = sum([l > t for l in decode_lengths])
            #tは0,1,2,3,...max(decode_lengths)まで動く decode_lengthsは降順であるため、はじめのbatch_sizeはsum([1,1,1,1,1,1])=6とかになる
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) 
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])) 
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds#t番目の単語の予測値を埋める
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

class Attention(nn.Module):
    """
    画像をよく表現する特徴量を得るためのエンコーダーモデル。
    
    Attributes
    ----------
    encoder_attention
        encoder_dim次元の入力をattention_dim次元に変換する全結合層。
    decoder_attention
        decoder_dim次元の入力をattention_dim次元に変換する全結合層。
    full_attention 
        attention_dim次元の入力を1次元に変換する全結合層。
    relu
        活性化関数ReLU。
    softmax
        活性化関数Softmax。
    """
    
    def __init__(self, encoder_dim : int, decoder_dim : int, attention_dim : int) -> None: 
        """ 
        用いる次元数の初期化をする。

        Parameters
        --------------
        encoder_dim : int
            エンコーダーの次元数
        decoder_dim : int
            デコーダーの次元数
        attention_dim : int
            アテンションの次元数
        """
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out,
                decoder_hidden):
        """ 
        順伝播を行う。

        Parameters
        --------------
        encoder_out : TensorType["batch", "width", "height", "encoder_dim" float]
            エンコードーの出力。
        decoder_hidden : TensorType["batch", "scores", "decoder_dim", float]
            デコーダーの隠れ層。

        Returns
        ---------
        attention_weighted_encoding : TensorType["batch", "width", "height", "encoder_dim", float]
            アテンション層の出力。
        alpha : TensorType["batch", "width", "height", "alpha":1, float]]
            アテンションの配列。

        """
        att1 = self.encoder_attention(encoder_out)
        att2 = self.decoder_attention(decoder_hidden)
        att = self.full_attention(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        #alpha (batch_size,width_of_image, height_of_image) encoder_out #(batch_size, width_of_image, height_of_image,encoder_dim)
        return attention_weighted_encoding, alpha

# from torchvision.models.efficientnet import efficientnet_v2_m
class Encoder(nn.Module):
    """
    画像をよく表現する特徴量を得るためのエンコーダーモデル。
    
    Attributes
    ----------
    split : str
        TRAINなら学習、VALなら検証、TESTならテストのデータセットであることを示す。
    h
        データを保存したHDF5ファイルのインスタンス。
    imgs : array-like
        画像の配列。画像の型はndarray。
    """
    def __init__(self, encoded_image_size : int =14) -> None:
        """ 
        モデルの初期化。

        Parameters
        --------------
        encoded_image_size : int
            特徴量マップのサイズ
        """
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        efficientnet_v2 = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT) 

        modules = list(efficientnet_v2.children())[:-2]
        self.efficientnet_v2 = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """ 
        順伝播を行う。

        Parameters
        --------------
        images : list[int, ndarray]
            バッチ毎の画像を保存。
        """
        out = self.efficientnet_v2(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)#batch_size, width_of_feature_map, height_of_feature_map,encoder_dim
        return out

    def fine_tune(self, fine_tune : bool =True) -> None:
        """ 
        fine_tuneがTrueならファインチューニングを行う。

        Parameters
        --------------
        fine_tune : bool
            Trueならファインチューニングを行い、Falseなら行わない。
        """
        
        
    
        for p in self.efficientnet_v2.parameters():
            p.requires_grad = False
            
        for i, c  in enumerate(list(self.efficientnet_v2.children())):
            if i > 700:
                for p in c:
                    p.requires_grad = True

            
        
         #for v2_m
#         for i, c  in enumerate(list(self.efficientnet_v2[0].children())):
#             if i > 4:
#               for p in c:
#                 p.requires_grad = True

