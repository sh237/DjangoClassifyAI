import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
from PIL import Image
import io, base64
import numpy as np
from io import BytesIO
import skimage.transform

#プロットしたグラフを画像データとして出力するための関数
def Output_Graph():
	buffer = BytesIO()                   #バイナリI/O(画像や音声データを取り扱う際に利用)
	plt.savefig(buffer, format="png")    #png形式の画像データを取り扱う
	buffer.seek(0)                       #ストリーム先頭のoffset byteに変更
	img   = buffer.getvalue()            #バッファの全内容を含むbytes
	graph = base64.b64encode(img)        #画像ファイルをbase64でエンコード
	graph = graph.decode("utf-8")        #デコードして文字列から画像に変換
	buffer.close()
	return graph

#グラフをプロットするための関数
def visualize_att(image, seq, alphas, rev_word_map, smooth=True):
    """
    画像とattentionの重みを可視化する

    Parameters
    ----------
    image_path : str
        画像のパス
    seq : list
        生成されたcaption
    alphas : list
        attentionの重み
    rev_word_map : dict
        IDと単語の対応表
    smooth : bool, optional
        attentionの重みをスムージングするかどうか
    """
    img_data = image.read()
    img_bin = io.BytesIO(img_data)
    image = Image.open(img_bin)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
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
    graph = Output_Graph()           #グラフプロット
    return graph