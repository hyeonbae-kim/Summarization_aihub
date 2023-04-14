# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

import collections
import logging
import unicodedata
import six
import os
from shutil import copyfile
from typing import List

from transformers import BertTokenizer, WordpieceTokenizer


logger = logging.getLogger(__name__)

try:
    from konlpy.tag import Mecab
except:
    logger.warning("Could not find Mecab installation! (ignore this if you are not using mecab)")


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
SPACE_PREFIX_4_MECAB = "[SP]"
JOIN_PREFIX_4_MECAB = "[J]"


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def strip_accents(text):
    """Strips accents from a piece of text."""
    # NFD (Normalization Form D) : 코드를 정준 분해
    #  발음 구별 기호가 붙은 글자가 하나로 처리되어 있을 경우, 이를 기호별로 나눈다.
    #  한글을 한글 음절 영역으로 썼을 경우, 이를 첫가끝 코드로 나눈다. (초성, 중성, 종성으로 분해)
    #  표준과 다른 조합 순서를 제대로 정렬한다.
    orig_text_len = len(text)
    normalized_text = unicodedata.normalize("NFD", text)
    output = []
    for char in normalized_text:
        cat = unicodedata.category(char)
        # Mn : Mark, nonspacing
        if cat == "Mn":
            continue
        output.append(char)

    # NFC (Normalization Form C) : 코드를 정준 분해한 뒤에, 다시 정준 결합
    #  발음 구별 기호가 붙었을 경우, 이를 코드 하나로 처리
    #  한글을 첫가끝 코드로 썼을 경우, 한글 음절 영역으로 처리
    converted = "".join(output)
    stripped = unicodedata.normalize("NFC", converted)

    return stripped if orig_text_len == len(stripped) else text


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BertKoreanMecabTokenizer(BertTokenizer):
    """BERT tokenizer for Korean text"""

    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        bos_token="<S>",
        eos_token="[EOS]",
        **kwargs
    ):
        """Constructs a MecabBertTokenizer.
        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
        """
        super(BertTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        self.vocab_path = vocab_file
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer()
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=unk_token)
        # mecab options
        self.pos_length = kwargs.get("pos_length", 2)
        self.spacing = kwargs.get("spacing", False)
        self.joining = kwargs.get("joining", False)

    def tokenize(self, text, mode='full', basic_tokens=None, **kwargs):
        """mecab 형태소 분석기를 이용한 토크나이저
        Args:
          text: 토크나이저 처리할 텍스트
          mode: basic, wordpiece, full(basic, wordpiece)
          basic_tokens: wordpiece 모드일때 이미 basic 모드 처리된 basic_tokens 사용
        Returns:
          토큰 리스트
        """
        if mode in ('basic', 'full'):
            basic_tokens, morp_list = self.basic_tokenizer.tokenize(text,
                                                                    pos_length=self.pos_length,
                                                                    spacing=self.spacing,
                                                                    joining=self.joining)
            if mode == 'basic':
                return (basic_tokens, morp_list)
        elif mode in ('wordpiece'):
            if not isinstance(basic_tokens, list):
                raise ValueError(f"basic_tokens should be a list in this mode[{mode}]!")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        split_tokens = []
        if mode in ('wordpiece', 'full'):
            for token in basic_tokens:
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)

        return split_tokens

    def detokenize(self, token_list):
        text = " ".join(token_list)

        # wordpieced(##) -> basic tokens
        text = text.replace(' ##', '')
        text = text.replace('##', '')
        detokenized = [token[:token.rfind('/')] if token.rfind('/') != -1 else token for token in text.split()]

        if self.spacing:
            detokenized = [
                token.replace(SPACE_PREFIX_4_MECAB, "", 1) + ' ' if token.startswith(SPACE_PREFIX_4_MECAB) else token
                for token in detokenized]
        elif self.joining:
            detokenized = [
                token.replace(JOIN_PREFIX_4_MECAB, "", 1) if token.startswith(JOIN_PREFIX_4_MECAB) else ' ' + token
                for token in detokenized]
        return "".join(detokenized).strip()

    # def convert_tokens_to_ids(self, tokens):
    #     return convert_by_vocab(self.vocab, tokens)

    # def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
    #     return convert_by_vocab(self.inv_vocab, ids)

    def strip_accents(self, text):
        return strip_accents(text)

    # TODO(volker): We simply copy the old vocab file to avoid saving a wrong vocab file in case of duplicate entries.
    #  Switch to BertTokenizer's `save_vocabulary` once we have a vocab without duplicate entries.
    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path

        copyfile(self.vocab_path, vocab_file)

        return (vocab_file,)


class BasicTokenizer:
    """BasicTokenizer는 형태소 분석기 처리를 포함한다.
        형분기는 최소 음절 단위로 분리
    """

    def __init__(self):
        try:
            self.mecab = Mecab()
        except:
            logging.warning("could not init mecab tokenizer (ignore this if you are not using mecab)")
            self.mecab = None

    def tokenize(self, text, pos_length=2, spacing=False, joining=False):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # 모든 문자 소문자 처리.
        text = text.lower()
        # u\2088 제거.
        text = text.replace("\u2028", "")
        text = strip_accents(text)

        # mecab flatten=True
        pos_list_: List[tuple] = self.mecab.pos(text, flatten=True)

        # ('영치기 영차', 'IC') -> ('영치기', IC') ('영차', 'IC')
        pos_list = []
        for morp, tag in pos_list_:
            for mor in morp.strip().split(' '):
                pos_list.append((mor, tag))

        morp_list: List[str] = [morp for morp, tag in pos_list if morp != '']

        if spacing or joining:
            tag_list: List[str] = [tag for morp, tag in pos_list if morp != '']
            word_lens: List[int] = [len(word) for word in text.split()]
            morp_lens_sum = sum([len(morp) for morp in morp_list])

            if sum(word_lens) != morp_lens_sum:
                raise ValueError(
                    f"Sum mismatched! word_lens_sum: {sum(word_lens)}, morp_lens_sum: {morp_lens_sum}, text: {text}")

            current_idx = 0
            total_idx = len(morp_list)
            output_tokens = []
            for word_len in word_lens:
                eojeol_output_tokens = []
                current_sum = 0
                while current_sum < word_len:
                    morp = morp_list[current_idx]
                    tag = tag_list[current_idx]
                    eojeol_output_tokens.append(morp.strip() + "/" + tag[:pos_length])
                    current_sum += len(morp)
                    current_idx += 1
                if spacing:
                    eojeol_output_tokens[-1] = SPACE_PREFIX_4_MECAB + eojeol_output_tokens[-1]
                elif joining:
                    eojeol_output_tokens = [eojeol_output_tokens[0]] + [JOIN_PREFIX_4_MECAB + token for token in eojeol_output_tokens[1:]]
                output_tokens.extend(eojeol_output_tokens)
                if current_idx == total_idx:
                    break

        else:
            output_tokens: List[str] = [morp + '/' + tag[:pos_length] for morp, tag in pos_list if morp != '']

        return (output_tokens, morp_list)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
