from django.db import models

import json
import easydict
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
# import Image
import torch.nn.functional as F
# from . import graph as graph_module
# from . import ml_models
import json
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from PIL import Image
from  typing import Tuple
from .ml_models import Encoder, DecoderWithAttention, Attention
import io, base64
from googletrans import Translator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import skimage.transform
Image.LOAD_TRUNCATED_IMAGES = True


class Photo(models.Model):
    image = models.ImageField(upload_to='photos')
    
    def predict(self):
        args = easydict.EasyDict(
        {
            # "img": '/content/drive/MyDrive/multimodalAI/image.png',
            "model": 'image2text/ml_models/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar',
            "word_map": 'image2text/ml_models/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
            "beam_size": 5,
            "smooth": 'smooth'
        }
        )
        print("predict started")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        with open(args.word_map, 'r') as j:
            word_map = json.load(j)
        print("word_map loaded")
        checkpoint = torch.load(args.model, map_location=str(device))
        print("model loaded")

        # decoder = DecoderWithAttention()
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        # encoder = Encoder()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()

        rev_word_map = {v: k for k, v in word_map.items()}

        seq, alphas = self.caption_image_beam_search(encoder, decoder, word_map, args.beam_size, device)
        alphas = torch.FloatTensor(alphas)
        image = self.show_image()
        graph = self.visualize_att(seq, alphas, rev_word_map)
        print(seq)
        sentence = [rev_word_map[ind] for ind in seq]
        sentence = sentence[1:-1]
        sentence = ' '.join(sentence)
        translator = Translator()
        translation = translator.translate(sentence, dest='ja')
        return sentence, translation.text, graph, image

    def Output_Graph(self):
        buffer = BytesIO()                   #バイナリI/O(画像や音声データを取り扱う際に利用)
        plt.savefig(buffer, format="png")    #png形式の画像データを取り扱う
        buffer.seek(0)                       #ストリーム先頭のoffset byteに変更
        img   = buffer.getvalue()            #バッファの全内容を含むbytes
        graph = base64.b64encode(img)        #画像ファイルをbase64でエンコード
        graph = graph.decode("utf-8")        #デコードして文字列から画像に変換
        buffer.close()
        return graph

    #グラフをプロットするための関数
    def visualize_att(self, seq, alphas, rev_word_map, smooth=True):
        """
        画像とattentionの重みを可視化する

        Parameters
        ----------
        seq : list
            生成されたcaption
        alphas : list
            attentionの重み
        rev_word_map : dict
            IDと単語の対応表
        smooth : bool, optional
            attentionの重みをスムージングするかどうか
        """
        plt.clf()
        img_data = self.image.read()
        # print("img_data",img_data)
        img_bin = io.BytesIO(img_data)
        print("img_bin", img_bin)
        image = Image.open(self.image)
        # print("img", img)

        image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

        words = [rev_word_map[ind] for ind in seq]
        print("lenwords",len(words))
        for t in range(len(words)):
            plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
            plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[t, :]
            if smooth:
                alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
            else:
                alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        # plt.show()
        graph = self.Output_Graph()           #グラフプロット
        return graph
    
    def show_image(self):
        plt.clf()
        image = Image.open(self.image)
        image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
        plt.axis("off")
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.imshow(image)
        graph = self.Output_Graph()
        return graph

    def caption_image_beam_search(self, encoder, decoder, word_map : dict, beam_size : int=3, device : str='cuda'):
        """
        画像に対し、beam searchを用いてcaptionを生成する処理。

        Parameters
        ----------
        encoder : EncoderCNN
            画像を特徴量に変換するモデル
        decoder : DecoderRNN
            特徴量を受け取り、captionを生成するモデル
        word_map : dict
            単語とIDの対応表
        beam_size : int, optional
            beam searchのサイズ
        device : torch.device
            モデルを動かすデバイス
        
        Returns
        -------
        seq : list
            生成されたcaption
        alphas : list
            attentionの重み
        """
        k = beam_size
        vocab_size = len(word_map)

        img_data = self.image.read()
        # print("img_data-------------------", img_data)
        img_bin = io.BytesIO(img_data)
        img = Image.open(img_bin).convert('RGB')
        img = img.resize([256, 256], Image.LANCZOS)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)

        image = image.unsqueeze(0)
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)

        seqs = k_prev_words

        top_k_scores = torch.zeros(k, 1).to(device)

        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            
            embeddings = decoder.embedding(k_prev_words).squeeze(1)

            awe, alpha = decoder.attention(encoder_out, h)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)

            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = (top_k_words / vocab_size).long()
            next_word_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                dim=1)
            
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas




