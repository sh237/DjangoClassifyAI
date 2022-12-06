from django.db import models
import re
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW,get_linear_schedule_with_warmup
CODE_PATTERN = re.compile(r"```.*?```", re.MULTILINE | re.DOTALL)
LINK_PATTERN = re.compile(r"!?\[([^\]\)]+)\]\([^\)]+\)")
IMG_PATTERN = re.compile(r"<img[^>]*>")
URL_PATTERN = re.compile(r"(http|ftp)s?://[^\s]+")
NEWLINES_PATTERN = re.compile(r"(\s*\n\s*)+")

class Text(models.Model):
    text = models.TextField()

    IMAGE_SIZE = 224
    MODEL_PATH = 'summarization/ml_models/'
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


    def predict(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            self.trained_model.cuda()

        self.preprocess_questionnaire_body(self.text)
        # return self.classes[predicted], percentage
        MAX_SOURCE_LENGTH = 512  # 入力文の最大トークン数
        MAX_TARGET_LENGTH = 126   # 生成文の最大トークン数

        # Pytorchのモデルを訓練モードから評価モードに変更
        self.trained_model.eval()

        # 入力文の前処理とトークナイズを行う
        inputs = [self.preprocess_questionnaire_body(self.text)]
        batch = self.tokenizer.batch_encode_plus(
            inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, 
            padding="longest", return_tensors="pt")

        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        # 出力文の生成処理を行う
        outputs = self.trained_model.generate(
            input_ids=input_ids, attention_mask=input_mask, 
            max_length=MAX_TARGET_LENGTH,
            return_dict_in_generate=True, output_scores=True,
            temperature=1.0,            # 生成にランダム性を入れる温度パラメータ
            num_beams=10,               # ビームサーチの探索幅
            diversity_penalty=1.0,      # 生成結果の多様性を生み出すためのペナルティ
            num_beam_groups=10,         # ビームサーチのグループ数
            num_return_sequences=5,    # 生成する文の数
            repetition_penalty=1.5,     # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

        # 生成されたトークンIDを文字列に変換する
        generated_sentences = [self.tokenizer.decode(ids, skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=False) 
                            for ids in outputs.sequences]

        # 生成された文案を表示する
        summary = []
        for i, sentence in enumerate(generated_sentences):
            if i == 0:
                summary.append(self.preprocess_questionnaire_body(sentence))
        return self.text, summary

    def unicode_normalize(self, cls, s):
        pt = re.compile('([{}]+)'.format(cls))

        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c

        s = ''.join(norm(x) for x in re.split(pt, s))
        s = re.sub('-', '-', s)
        return s

    def remove_extra_spaces(self, s):
        s = re.sub('[  ]+', ' ', s)
        blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                        '\u3040-\u309F',  # HIRAGANA
                        '\u30A0-\u30FF',  # KATAKANA
                        '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                        '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                        ))
        basic_latin = '\u0000-\u007F'

        def remove_space_between(cls1, cls2, s):
            p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
            while p.search(s):
                s = p.sub(r'\1\2', s)
            return s

        s = remove_space_between(blocks, blocks, s)
        s = remove_space_between(blocks, basic_latin, s)
        s = remove_space_between(basic_latin, blocks, s)
        return s

    def normalize_neologd(self, s):
        s = s.strip()
        s = self.unicode_normalize('0-9A-Za-ｚ｡-ﾟ', s)

        def maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
        s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
        s = s.translate(
            maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

        s = self.remove_extra_spaces(s)
        s = self.unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        return s

    def clean_markdown(self,markdown_text):
        markdown_text = CODE_PATTERN.sub(r"", markdown_text)
        markdown_text = LINK_PATTERN.sub(r"\1", markdown_text)
        markdown_text = IMG_PATTERN.sub(r"", markdown_text)
        markdown_text = URL_PATTERN.sub(r"", markdown_text)
        markdown_text = NEWLINES_PATTERN.sub(r"\n", markdown_text)
        markdown_text = markdown_text.replace("`", "")
        return markdown_text

    def normalize_text(self, markdown_text):
        markdown_text = self.clean_markdown(markdown_text)
        markdown_text = markdown_text.replace("\t", " ")
        markdown_text = self.normalize_neologd(markdown_text).lower()
        markdown_text = markdown_text.replace("\n", " ")
        return markdown_text

    def preprocess_questionnaire_body(self,markdown_text):
        return "body: " + self.normalize_text(markdown_text)[:4000]

    def postprocess_abstract(self,abstract):
        return re.sub(r"abstract: ", "", abstract)