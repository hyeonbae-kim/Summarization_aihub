import torch
import argparse
from transformers import BertTokenizer
from pydantic import BaseModel
from typing import List, Union
# from kamino import ProdModel, Kamino
# from starlette.responses import FileResponse
from kss import split_sentences
import gradio as gr

from tokenization import BertKoreanMecabTokenizer
from models.model_builder import AbsSummarizer, ExtSummarizer
from models.predictor import build_predictor
from models.trainer_ext import build_trainer
from prepro.split_sentence import text_processing
from others.logging import def_logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class NetInput(BaseModel):
    article: str


class Data(BaseModel):
    net_input: List[NetInput]

    extractive = False
    top_k = 3

    def __len__(self):
        return len(self.net_input)

    def __getitem__(self, val):
        return self.copy(update={"net_input": self.net_input[val]})


class Summary(BaseModel):
    summaries: Union[List[List[str]], List[str]]
    #                 Extractive      Abstractive

    def __add__(self, x):
        self.summaries.extend(x.summaries)
        return self


# class Presumm(ProdModel):
class Presumm():
    input_model = Data
    output_model = Summary

    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if args.tokenizer == "multi":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        elif args.tokenizer == "mecab":
            self.tokenizer = BertKoreanMecabTokenizer(args.vocab, joining=True)
        self.symbols = {'BOS': self.tokenizer.vocab[args.tgt_bos], 'EOS': self.tokenizer.vocab[args.tgt_eos],
                   'PAD': self.tokenizer.vocab['[PAD]'], 'EOQ': self.tokenizer.vocab[args.tgt_sent_split]}

        self.predictor = {
            "abs": self._get_predictor(args),
            "ext": self._get_predictor(args, ext=True)
        }

        self.max_pos = args.max_pos

        def_logger.info(f'*** Model Info ***')
        def_logger.info(f"\tdevice: {self.device}")
        def_logger.info(f"\tabs model: {self.predictor.get('abs').model.__class__.__name__ if self.predictor.get('abs') is not None else 'None'}")
        def_logger.info(f"\text model: {self.predictor.get('ext').model.__class__.__name__ if self.predictor.get('ext') is not None else 'None'}")
        def_logger.info(f"\ttokenizer: {self.tokenizer.__class__.__name__}")

    def _get_predictor(self, args, ext=False):
        if args.ext_model and ext:
            checkpoint = torch.load(args.ext_model, map_location=lambda storage, loc: storage)
            model = ExtSummarizer(args, self.device, checkpoint)
            predictor = build_trainer(args, model=model, tokenizer=self.tokenizer)
        elif args.abs_model and not ext:
            checkpoint = torch.load(args.abs_model, map_location=lambda storage, loc: storage)
            model = AbsSummarizer(args, self.device, checkpoint)
            predictor = build_predictor(args, self.tokenizer, self.symbols, model)
        else:
            return None

        return predictor

    def predict(self, data):
        # texts = [ni.article for ni in data.net_input]
        texts = [ni for ni in data]
        input_tokens = []
        sentences_batch = []
        for text in texts:
            # Clean data
            if isinstance(self.tokenizer, BertKoreanMecabTokenizer):
                text = text_processing(text)

            # Split sentences
            sentences = [sentence for sentence in split_sentences(text)]
            sentences_batch.append(sentences)

            # Insert [SEP], [CLS] tokens between sentences
            tokens = ' {} {} '.format(self.tokenizer.sep_token, self.tokenizer.cls_token).join(
                [" ".join(self.tokenizer.tokenize(sentence)) for sentence in sentences])

            # Add [CLS] to start of tokens and add [SEP] to end of tokens
            tokens = [self.tokenizer.cls_token] + tokens.split() + [self.tokenizer.sep_token]

            input_tokens.append(tokens)

        # Pad tokens
        max_len = max([len(tokens) for tokens in input_tokens])
        max_len = self.max_pos if max_len > self.max_pos else max_len

        # Convert tokens to ids
        inputs = []
        for tokens in input_tokens:
            pad = ['[PAD]'] * (max_len - len(tokens))
            tokens = tokens[:max_len][:-1] + [tokens[-1]] + pad
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            inputs.append(input_ids)

        # Get prediction
        def_logger.debug(f'data: {data}, inputs: {inputs}, sentences_batch: {sentences_batch}')
        # if data.extractive:
            # pred = self.predictor["ext"].predict(inputs, sentences_batch, n_sentence=data.top_k)
        # else:
            # pred = self.predictor["abs"].predict(inputs)
        pred = self.predictor["abs"].predict(inputs)

        return Summary(summaries=pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument("-abs_model", default='/content/Summarization_aihub/models/MultiSumAbs_report_512/model_step_20000.pt', type=str)
    parser.add_argument("-ext_model", default='', type=str)
    parser.add_argument("-checkpoint_path", default='')
    parser.add_argument("-temp_dir", default='../temp_dir')

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-max_pos", default=1024, type=int)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=210, type=int)

    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-tokenizer", default='multi', type=str, choices=['multi', 'mecab'])
    parser.add_argument("-vocab", default='', type=str)

    parser.add_argument("-tgt_bos", default='[unused1]', type=str) #[rsvd2]
    parser.add_argument("-tgt_eos", default='[unused2]', type=str) #[rsvd3]
    parser.add_argument("-tgt_sent_split", default='[unused3]', type=str) #[rsvd4]

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    local_worker_devices = args.visible_gpus.split(",")
    output = Presumm(args)
    # k = Kamino(Presumm, local_worker_devices=local_worker_devices, frontdoor_port=3248, queue_port=3249, gather_port=3250,
              #  result_port=3251, skip_port=3252, control_port=3253, port=8001, batch_size=4,
              #  args=args)
    
    #print(output.predict(data='/content/Summarization_aihub/data/test/report_briefing.test.3.bert.pt'))
    

    demo = gr.Interface(fn=output.predict, inputs=gr.File(label='Input File'), outputs="text")
    demo.launch(share=True)

    # @k.app.get("/")
    # def index():
        # return FileResponse("/home/src/demo/index.html")

    # k.run()
