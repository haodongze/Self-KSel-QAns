"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
import numpy as np
import random

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.base_model import tile
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def have_ans(ans_all, doc):
    for ans in ans_all:
        if ans in doc:
            return True
    return False

@registry.register_model("blip2_t5_okvqa_select")
class Blip2T5_okvqa_select(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        knowledge_num=5,
        max_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        use_lora=True,
        task="train_qa",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        # self.tokenizer = self.init_tokenizer()

        if 'select' in task:
            self.visual_encoder, self.ln_vision, self.ln_vision_select = self.init_vision_encoder_okvqa(
                vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
        else:
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        if use_lora:
            for name, param in self.t5_model.named_parameters():
                param.data = param.data.bfloat16()
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            print_trainable_parameters(self.t5_model)
        else:
            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
                param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # Q-Former for Selector
        if 'select' in task:
            self.Qformer_select = self.Qformer
            self.query_tokens_select = self.query_tokens
            # self.Qformer_select, self.query_tokens_select = self.init_Qformer(
            #     num_query_token, self.visual_encoder.num_features)
            #
            # self.Qformer_select.cls = None
            # self.Qformer_select.bert.embeddings.word_embeddings = None
            # self.Qformer_select.bert.embeddings.position_embeddings = None
            # for layer in self.Qformer_select.bert.encoder.layer:
            #     layer.output = None
            #     layer.intermediate = None

            if use_lora:
                self.t5_model_select = T5ForConditionalGeneration.from_pretrained(
                    t5_model, config=t5_config
                )
                for name, param in self.t5_model_select.named_parameters():
                    param.data = param.data.bfloat16()
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
                )
                self.t5_model_select = get_peft_model(self.t5_model_select, lora_config)
                print_trainable_parameters(self.t5_model_select)
            self.t5_proj_select = nn.Linear(
                self.Qformer_select.config.hidden_size, self.t5_model.config.hidden_size
            )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        self.task = task

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

        self.knowledge_num = knowledge_num
        print(self.knowledge_num)
        # self.knowledge_prefix = ['Knowledge: ']
        # self._prefix = ['Knowledge {}: '.format(str(i + 1)) for i in range(knowledge_num)]
        self.yes_id, self.no_id = 4273, 150

        # if 'freeze_qa' in task:
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False
        # self.t5_proj.requires_grad = False

        # if 'freeze_select' in task:
        if 'select' in task:
            for name, param in self.Qformer_select.named_parameters():
                param.requires_grad = False
            self.query_tokens_select.requires_grad = False
            # self.t5_proj_select.requires_grad = False

    def forward(self, samples,
                use_nucleus_sampling=False,
                num_beams=5, max_length=30,
                min_length=1, top_p=0.9,
                repetition_penalty=1.0, length_penalty=1.0,
                num_captions=1, temperature=1,
                ):
        batch_size = samples["image"].size(0)
        if type(samples["passages"][0]) is list:
            n = len(samples["passages"][0])
            new_text_input = []
            new_passages = []
            new_gold_answer = []
            new_answers = []
            for b, passage in enumerate(samples["passages"]):
                new_passages.extend(passage)
                new_text_input.extend([samples["text_input"][b]] * n)
                new_gold_answer.extend([samples['gold_answer'][b]] * n)
                for i in range(n):
                    new_answers.append(samples['answers'][b])
            samples["passages"] = new_passages
            samples["text_input"] = new_text_input
            samples['gold_answer'] = new_gold_answer
            samples['answers'] = new_answers

        image = samples["image"]
        with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # Selector self-refinement
        if 'train_selector' in self.task:

            # ========= Generate pseudo labels by frozen answerer ============
            with torch.no_grad():

                image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
                image_embeds_ = self.ln_vision(image_embeds_) # bs,256,c

                query_tokens_qa = self.query_tokens.expand(image_embeds_.shape[0], -1, -1)
                query_output_qa = self.Qformer.bert(
                    query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds_,
                    encoder_attention_mask=image_atts_, return_dict=True)
                inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                inputs_t5_qa = torch.repeat_interleave(inputs_t5_qa, n, 0)
                atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)

                if self.prompt.count("{}") == 2:
                    text_input = [self.prompt.format(question, passage) for question, passage in
                                  zip(samples["text_input"], samples["passages"])]
                else:
                    text_input = [self.prompt.format(question) for question in samples["text_input"]]
                text_input_qa = text_input
                answer = samples['gold_answer']
                # ans_idx = [self.ANS_MAP[a[-1]] for a in answer]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # # Frame Prefix
                    # frame_prefix = self.t5_tokenizer(
                    #     self.frame_prefix, padding="longest", add_special_tokens=False,
                    #     truncation=True, max_length=self.max_txt_len, return_tensors="pt",
                    # ).to(image.device)  #
                    # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                    # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                    # Question, options input
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_qa = input_tokens_qa.input_ids
                    input_attention_mask_qa = input_tokens_qa.attention_mask
                    # input_ids_qa = torch.repeat_interleave(input_tokens_qa.input_ids, t, 0)
                    # input_attention_mask_qa = torch.repeat_interleave(input_tokens_qa.attention_mask, t, 0)

                    # Output target
                    output_tokens_qa = self.t5_tokenizer(
                        answer, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    targets_qa = output_tokens_qa.input_ids.masked_fill(
                        output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                    output_tokens_mask_qa = targets_qa
                    # output_tokens_mask_qa = torch.repeat_interleave(output_tokens_qa.attention_mask, t, dim=0)
                    # targets_qa = torch.repeat_interleave(targets_qa, t, dim=0)

                    # input for QA
                    # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_ids_qa)
                    inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([atts_t5_qa, input_attention_mask_qa], dim=1)
                    # inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                    # encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_attention_mask_qa], dim=1)

                    # outputs_embed_qa = self.t5_model(
                    #     inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    #     decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                    # pred_logits_qa = outputs_embed_qa.logits.detach()
                    # pred_logits_qa = pred_logits_qa[:, 1, self.answer_id]  # b*t, 5
                    # pred_ans = torch.argmax(pred_logits_qa, dim=-1)
                    # pred_ans = pred_ans.reshape(b, -1)  # b, t
                    # # print('pred_ans', pred_ans)
                    # pseudo_label = []
                    # for i, preds in enumerate(pred_ans):
                    #     for p in preds:
                    #         if p == ans_idx[i]:
                    #             pseudo_label.append('yes')
                    #         else:
                    #             pseudo_label.append('no')
                    outputs_qa = self.t5_model.generate(
                        inputs_embeds=inputs_embeds_qa,
                        attention_mask=encoder_atts_qa,
                        do_sample=False,
                        num_beams=num_beams,
                        max_new_tokens=10,
                        min_length=1,
                        length_penalty=1.0,
                    )
                    output_text_qa = self.t5_tokenizer.batch_decode(
                        outputs_qa, skip_special_tokens=True
                    )
                    pseudo_label = []
                    for i, pred in enumerate(output_text_qa):
                        # if pred.lower() == samples['gold_answer'][i].lower():
                        #     pseudo_label.append('yes')
                        # else:
                        #     pseudo_label.append('no')

                        # ans_all = [ans.lower() for ans in samples['answers'][i]]
                        # if pred.lower() in ans_all:
                        #     pseudo_label.append('yes')
                        # else:
                        #     pseudo_label.append('no')

                        ans_all = [ans.lower() for ans in samples['answers'][i]]
                        if pred.lower() == samples['gold_answer'][i].lower() and have_ans(ans_all, samples["passages"][
                            i].lower()):
                            pseudo_label.append('yes')
                        else:
                            pseudo_label.append('no')

            # ================================================================

            # ============== Train selector with pseudo labels =================
            text_input_select = [self.prompt.format(question, passage) for question, passage in
                                  zip(samples["text_input"], samples["passages"])]
            query_tokens_select = self.query_tokens_select.expand(image_embeds.shape[0], -1, -1)
            image_embeds = self.ln_vision_select(image_embeds)

            query_output_select = self.Qformer_select.bert(
                query_embeds=query_tokens_select, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)  # bn, 32, c
            inputs_t5_select = self.t5_proj_select(query_output_select.last_hidden_state)  # bn, 32, c
            inputs_t5_select = torch.repeat_interleave(inputs_t5_select, n, 0)
            atts_t5_select = torch.ones(inputs_t5_select.size()[:-1], dtype=torch.long).to(image.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # frame_prefix = self.t5_tokenizer(
                #     self.frame_prefix, padding="longest", add_special_tokens=False,
                #     truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)

                input_tokens_select = self.t5_tokenizer(
                    text_input_select, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                input_ids_select = input_tokens_select.input_ids
                input_attention_mask_select = input_tokens_select.attention_mask
                # input_ids_loc = torch.repeat_interleave(input_tokens_loc.input_ids, t, 0)
                # input_attention_mask_loc = torch.repeat_interleave(input_tokens_loc.attention_mask, t, 0)
                inputs_embeds_select = self.t5_model.encoder.embed_tokens(input_ids_select)

                inputs_embeds_select = torch.cat([inputs_t5_select, inputs_embeds_select], dim=1)
                encoder_atts_select = torch.cat([atts_t5_select, input_attention_mask_select], dim=1)
                # inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                # encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)

                output_tokens_select = self.t5_tokenizer(
                    pseudo_label, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                targets_select = output_tokens_select.input_ids.masked_fill(
                    output_tokens_select.input_ids == self.t5_tokenizer.pad_token_id, -100)
                output_tokens_select_mask = output_tokens_select.attention_mask

                outputs_select = self.t5_model(
                    inputs_embeds=inputs_embeds_select, attention_mask=encoder_atts_select,
                    decoder_attention_mask=output_tokens_select_mask,
                    return_dict=True, labels=targets_select)
                loss = outputs_select.loss

            return {"loss": loss}

        # Finetune answerer with selector
        elif 'train_qa_with_selector' in self.task:
            # knowledge selection
            with torch.no_grad():
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # b 256 c
                image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
                image_embeds_ = self.ln_vision_select(image_embeds_)

                text_input_select = [self.prompt.format(question, passage) for question, passage in
                                     zip(samples["text_input"], samples["passages"])]
                query_tokens_select = self.query_tokens_select.expand(image_embeds_.shape[0], -1, -1)
                query_output_select = self.Qformer_select.bert(
                    query_embeds=query_tokens_select, encoder_hidden_states=image_embeds_,
                    encoder_attention_mask=image_atts_, return_dict=True)
                inputs_t5_select = self.t5_proj_select(query_output_select.last_hidden_state)
                inputs_t5_select = torch.repeat_interleave(inputs_t5_select, n, 0)

                atts_t5_select = torch.ones(inputs_t5_select.size()[:-1], dtype=torch.long).to(image.device)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # frame_prefix = self.t5_tokenizer(
                    #     self.frame_prefix, padding="longest", add_special_tokens=False,
                    #     truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                    # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                    # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)

                    input_tokens_select = self.t5_tokenizer(
                        text_input_select, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_select = input_tokens_select.input_ids
                    input_attention_mask_select = input_tokens_select.attention_mask
                    # input_ids_select = torch.repeat_interleave(input_tokens_select.input_ids, t, 0)
                    # input_attention_mask_loc = torch.repeat_interleave(input_tokens_select.attention_mask, t, 0)
                    inputs_embeds_select = self.t5_model.encoder.embed_tokens(input_ids_select)
                    inputs_embeds_select = torch.cat([inputs_t5_select, inputs_embeds_select], dim=1)
                    encoder_atts_select = torch.cat([atts_t5_select, input_attention_mask_select], dim=1)
                    # inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                    # encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)

                    outputs_select = self.t5_model.generate(
                        inputs_embeds=inputs_embeds_select, attention_mask=encoder_atts_select,
                        do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature, num_beams=1,
                        max_new_tokens=max_length, min_length=min_length, repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty, num_return_sequences=num_captions,
                        return_dict_in_generate=True, output_hidden_states=True, output_scores=True)

                    pred_logits_select = outputs_select.scores[0]
                    select_yes = pred_logits_select[:, self.yes_id]
                    select_yes = select_yes.reshape(batch_size, -1)

            text_input_qa = [self.prompt.format(question, passage) for question, passage in
                                  zip(samples["text_input"], samples["passages"])]
            answer = samples["gold_answer"]

            select_knowledge_idx = torch.topk(select_yes, self.knowledge_num, dim=-1).indices.tolist()
            sorted_knowledge_idx = []
            for k in select_knowledge_idx:
                sorted_knowledge_idx.append(sorted(k))

            ## random select
            # sorted_knowledge_idx = [random.sample(range(30), 5) for j in range(batch_size)]

            select_text_input_qa = []
            select_answer = []
            for i, ks in enumerate(sorted_knowledge_idx):
                for j, k in enumerate(ks):
                    select_text_input_qa.append(text_input_qa[i*n+k])
                    select_answer.append(answer[i * n + k])

            image_embeds = self.ln_vision(image_embeds)  # b, 256,c
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # bt n c
            query_tokens_qa = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            inputs_t5_qa = torch.repeat_interleave(inputs_t5_qa, self.knowledge_num, 0)
            atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # vid_prefix = self.t5_tokenizer(
                #     self.vid_prefix, padding="longest", add_special_tokens=False,
                #     truncation=True, max_length=self.max_txt_len, return_tensors="pt", ).to(image.device)  #
                # vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                # vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                # vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id)  # b t n_word c

                # inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2)  # b, t, n_word + m, c
                # atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2)  # b, t, n_word + m
                # inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                # atts_t5_qa = atts_t5_qa.reshape(b, -1)

                input_tokens_qa = self.t5_tokenizer(
                    select_text_input_qa, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids)
                inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)

                output_tokens_qa = self.t5_tokenizer(
                    select_answer, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                targets_qa = output_tokens_qa.input_ids.masked_fill(
                    output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                output_tokens_mask_qa = output_tokens_qa.attention_mask

                outputs_qa = self.t5_model(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                loss = outputs_qa.loss

                return {"loss": loss}
        elif 'cycle' in self.task:
            ####### 训练 answerer #######
            # knowledge selection
            with torch.no_grad():
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # b 256 c
                image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
                image_embeds_ = self.ln_vision_select(image_embeds_)

                text_input_select = [self.prompt.format(question, passage) for question, passage in
                                     zip(samples["text_input"], samples["passages"])]
                query_tokens_select = self.query_tokens_select.expand(image_embeds_.shape[0], -1, -1)
                query_output_select = self.Qformer_select.bert(
                    query_embeds=query_tokens_select, encoder_hidden_states=image_embeds_,
                    encoder_attention_mask=image_atts_, return_dict=True)
                inputs_t5_select = self.t5_proj_select(query_output_select.last_hidden_state)
                inputs_t5_select = torch.repeat_interleave(inputs_t5_select, n, 0)

                atts_t5_select = torch.ones(inputs_t5_select.size()[:-1], dtype=torch.long).to(image.device)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # frame_prefix = self.t5_tokenizer(
                    #     self.frame_prefix, padding="longest", add_special_tokens=False,
                    #     truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                    # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                    # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)

                    input_tokens_select = self.t5_tokenizer(
                        text_input_select, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_select = input_tokens_select.input_ids
                    input_attention_mask_select = input_tokens_select.attention_mask
                    # input_ids_select = torch.repeat_interleave(input_tokens_select.input_ids, t, 0)
                    # input_attention_mask_loc = torch.repeat_interleave(input_tokens_select.attention_mask, t, 0)
                    inputs_embeds_select = self.t5_model_select.encoder.embed_tokens(input_ids_select)
                    inputs_embeds_select = torch.cat([inputs_t5_select, inputs_embeds_select], dim=1)
                    encoder_atts_select = torch.cat([atts_t5_select, input_attention_mask_select], dim=1)
                    # inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                    # encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)

                    outputs_select = self.t5_model_select.generate(
                        inputs_embeds=inputs_embeds_select, attention_mask=encoder_atts_select,
                        do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature, num_beams=1,
                        max_new_tokens=max_length, min_length=min_length, repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty, num_return_sequences=num_captions,
                        return_dict_in_generate=True, output_hidden_states=True, output_scores=True)

                    pred_logits_select = outputs_select.scores[0]
                    select_yes = pred_logits_select[:, self.yes_id]
                    select_yes = select_yes.reshape(batch_size, -1)

            select_knowledge_idx = torch.topk(select_yes, self.knowledge_num, dim=-1).indices.tolist()
            sorted_knowledge_idx = []
            for k in select_knowledge_idx:
                sorted_knowledge_idx.append(sorted(k))

            ### 分别输入 ###
            text_input_qa = [self.prompt.format(question, passage) for question, passage in
                             zip(samples["text_input"], samples["passages"])]
            answer = samples["gold_answer"]

            select_text_input_qa = []
            select_answer = []
            for i, ks in enumerate(sorted_knowledge_idx):
                for j, k in enumerate(ks):
                    select_text_input_qa.append(text_input_qa[i * n + k])
                    select_answer.append(answer[i * n + k])

            ### concat ###
            # select_question = []
            # select_passage = []
            # select_answer = []
            # for i, ks in enumerate(sorted_knowledge_idx):
            #     docs = []
            #     for j, k in enumerate(ks):
            #         docs.append(samples["passages"][i * n + k])
            #     select_answer.append(answer[i * n])
            #     select_question.append(samples["text_input"][i*n])
            #     select_passage.append(' '.join(docs))
            # select_text_input_qa = [self.prompt.format(question, passage) for question, passage in
            #                  zip(select_question, select_passage)]

            image_embeds = self.ln_vision(image_embeds)  # b, 256,c
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # bt n c
            query_tokens_qa = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            inputs_t5_qa = torch.repeat_interleave(inputs_t5_qa, self.knowledge_num, 0)
            atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # vid_prefix = self.t5_tokenizer(
                #     self.vid_prefix, padding="longest", add_special_tokens=False,
                #     truncation=True, max_length=self.max_txt_len, return_tensors="pt", ).to(image.device)  #
                # vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                # vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                # vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id)  # b t n_word c

                # inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2)  # b, t, n_word + m, c
                # atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2)  # b, t, n_word + m
                # inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                # atts_t5_qa = atts_t5_qa.reshape(b, -1)

                input_tokens_qa = self.t5_tokenizer(
                    select_text_input_qa, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids)
                inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)

                output_tokens_qa = self.t5_tokenizer(
                    select_answer, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                targets_qa = output_tokens_qa.input_ids.masked_fill(
                    output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                output_tokens_mask_qa = output_tokens_qa.attention_mask

                outputs_qa = self.t5_model(
                    inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                loss_qa = outputs_qa.loss
            ####### 训练 selector #######
            # ========= Generate pseudo labels by frozen answerer ============
            with torch.no_grad():
                image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
                image_embeds_ = self.ln_vision(image_embeds_)  # bs,256,c

                query_tokens_qa = self.query_tokens.expand(image_embeds_.shape[0], -1, -1)
                query_output_qa = self.Qformer.bert(
                    query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds_,
                    encoder_attention_mask=image_atts_, return_dict=True)
                inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
                inputs_t5_qa = torch.repeat_interleave(inputs_t5_qa, n, 0)
                atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)

                if self.prompt.count("{}") == 2:
                    text_input = [self.prompt.format(question, passage) for question, passage in
                                  zip(samples["text_input"], samples["passages"])]
                else:
                    text_input = [self.prompt.format(question) for question in samples["text_input"]]
                text_input_qa = text_input
                answer = samples['gold_answer']
                # ans_idx = [self.ANS_MAP[a[-1]] for a in answer]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # # Frame Prefix
                    # frame_prefix = self.t5_tokenizer(
                    #     self.frame_prefix, padding="longest", add_special_tokens=False,
                    #     truncation=True, max_length=self.max_txt_len, return_tensors="pt",
                    # ).to(image.device)  #
                    # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                    # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                    # Question, options input
                    input_tokens_qa = self.t5_tokenizer(
                        text_input_qa, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    input_ids_qa = input_tokens_qa.input_ids
                    input_attention_mask_qa = input_tokens_qa.attention_mask
                    # input_ids_qa = torch.repeat_interleave(input_tokens_qa.input_ids, t, 0)
                    # input_attention_mask_qa = torch.repeat_interleave(input_tokens_qa.attention_mask, t, 0)

                    # Output target
                    output_tokens_qa = self.t5_tokenizer(
                        answer, padding="longest", truncation=True,
                        max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    targets_qa = output_tokens_qa.input_ids.masked_fill(
                        output_tokens_qa.input_ids == self.t5_tokenizer.pad_token_id, -100)
                    output_tokens_mask_qa = targets_qa
                    # output_tokens_mask_qa = torch.repeat_interleave(output_tokens_qa.attention_mask, t, dim=0)
                    # targets_qa = torch.repeat_interleave(targets_qa, t, dim=0)

                    # input for QA
                    # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)
                    inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_ids_qa)
                    inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                    encoder_atts_qa = torch.cat([atts_t5_qa, input_attention_mask_qa], dim=1)
                    # inputs_embeds_qa = torch.cat([frame_predix_embed, inputs_t5_qa, inputs_embeds_qa], dim=1)
                    # encoder_atts_qa = torch.cat([frame_prefix_mask, atts_t5_qa, input_attention_mask_qa], dim=1)

                    # outputs_embed_qa = self.t5_model(
                    #     inputs_embeds=inputs_embeds_qa, attention_mask=encoder_atts_qa,
                    #     decoder_attention_mask=output_tokens_mask_qa, return_dict=True, labels=targets_qa)
                    # pred_logits_qa = outputs_embed_qa.logits.detach()
                    # pred_logits_qa = pred_logits_qa[:, 1, self.answer_id]  # b*t, 5
                    # pred_ans = torch.argmax(pred_logits_qa, dim=-1)
                    # pred_ans = pred_ans.reshape(b, -1)  # b, t
                    # # print('pred_ans', pred_ans)
                    # pseudo_label = []
                    # for i, preds in enumerate(pred_ans):
                    #     for p in preds:
                    #         if p == ans_idx[i]:
                    #             pseudo_label.append('yes')
                    #         else:
                    #             pseudo_label.append('no')

                    ###########根据生成的答案来打伪标签##############
                    outputs_qa = self.t5_model.generate(
                        inputs_embeds=inputs_embeds_qa,
                        attention_mask=encoder_atts_qa,
                        do_sample=False,
                        num_beams=num_beams,
                        max_new_tokens=10,
                        min_length=1,
                        length_penalty=1.0,
                    )
                    output_text_qa = self.t5_tokenizer.batch_decode(
                        outputs_qa, skip_special_tokens=True
                    )
                    pseudo_label = []
                    for i, pred in enumerate(output_text_qa):
                        # if pred.lower() == samples['gold_answer'][i].lower():
                        #     pseudo_label.append('yes')
                        # else:
                        #     pseudo_label.append('no')

                        # ans_all = [ans.lower() for ans in samples['answers'][i]]
                        # if pred.lower() in ans_all:
                        #     pseudo_label.append('yes')
                        # else:
                        #     pseudo_label.append('no')

                        ans_all = [ans.lower() for ans in samples['answers'][i]]
                        if pred.lower() == samples['gold_answer'][i].lower() and have_ans(ans_all, samples["passages"][i].lower()):
                            pseudo_label.append('yes')
                        else:
                            pseudo_label.append('no')

                    ###########根据TIE来打伪标签##############
                    # outputs_with_k = self.t5_model.generate(
                    #     inputs_embeds=inputs_embeds_qa,
                    #     attention_mask=encoder_atts_qa,
                    #     do_sample=False,
                    #     num_beams=num_beams,
                    #     max_new_tokens=10,
                    #     min_length=1,
                    #     length_penalty=1.0,
                    #     return_dict_in_generate=True,
                    #     output_scores=True,
                    # )
                    # output_text_with_k = self.t5_tokenizer.batch_decode(
                    #     outputs_with_k['sequences'], skip_special_tokens=True
                    # )
                    #
                    # text_input = ["Question: {} \n Answer:".format(question) for question in samples["text_input"]]
                    # input_tokens = self.t5_tokenizer(
                    #     text_input, padding="longest", truncation=True,
                    #     max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                    # input_ids = input_tokens.input_ids
                    # input_attention_mask = input_tokens.attention_mask
                    # inputs_embeds = self.t5_model.encoder.embed_tokens(input_ids)
                    # inputs_embeds = torch.cat([inputs_t5_qa, inputs_embeds], dim=1)
                    # encoder_atts = torch.cat([atts_t5_qa, input_attention_mask], dim=1)
                    # outputs_wo_k = self.t5_model.generate(
                    #     inputs_embeds=inputs_embeds,
                    #     attention_mask=encoder_atts,
                    #     do_sample=False,
                    #     num_beams=num_beams,
                    #     max_new_tokens=10,
                    #     min_length=1,
                    #     length_penalty=1.0,
                    #     return_dict_in_generate=True,
                    #     output_scores=True,
                    # )
                    # output_text_wo_k = self.t5_tokenizer.batch_decode(
                    #     outputs_wo_k['sequences'], skip_special_tokens=True
                    # )
                    # tie = np.exp(outputs_with_k['sequences_scores'].cpu().numpy())-np.exp(outputs_wo_k['sequences_scores'].cpu().numpy())
                    # pseudo_label = []
                    # for i, pred in enumerate(output_text_with_k):
                    #     ans_all = [ans.lower() for ans in samples['answers'][i]]
                    #     if pred.lower() == samples['gold_answer'][i].lower():
                    #         if output_text_wo_k[i].lower() == samples['gold_answer'][i].lower():
                    #             if tie[i] < 0.1:
                    #                 pseudo_label.append('no')
                    #             else:
                    #                 pseudo_label.append('yes')
                    #         elif have_ans(ans_all, samples["passages"][i].lower()):
                    #             pseudo_label.append('yes')
                    #         else:
                    #             pseudo_label.append('no')
                    #     else:
                    #         pseudo_label.append('no')


                # ================================================================

                # ============== Train selector with pseudo labels =================
            text_input_select = [self.prompt.format(question, passage) for question, passage in
                                 zip(samples["text_input"], samples["passages"])]
            query_tokens_select = self.query_tokens_select.expand(image_embeds.shape[0], -1, -1)
            image_embeds = self.ln_vision_select(image_embeds)

            query_output_select = self.Qformer_select.bert(
                query_embeds=query_tokens_select, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)  # bn, 32, c
            inputs_t5_select = self.t5_proj_select(query_output_select.last_hidden_state)  # bn, 32, c
            inputs_t5_select = torch.repeat_interleave(inputs_t5_select, n, 0)
            atts_t5_select = torch.ones(inputs_t5_select.size()[:-1], dtype=torch.long).to(image.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # frame_prefix = self.t5_tokenizer(
                #     self.frame_prefix, padding="longest", add_special_tokens=False,
                #     truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)

                input_tokens_select = self.t5_tokenizer(
                    text_input_select, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                input_ids_select = input_tokens_select.input_ids
                input_attention_mask_select = input_tokens_select.attention_mask
                # input_ids_loc = torch.repeat_interleave(input_tokens_loc.input_ids, t, 0)
                # input_attention_mask_loc = torch.repeat_interleave(input_tokens_loc.attention_mask, t, 0)
                inputs_embeds_select = self.t5_model_select.encoder.embed_tokens(input_ids_select)

                inputs_embeds_select = torch.cat([inputs_t5_select, inputs_embeds_select], dim=1)
                encoder_atts_select = torch.cat([atts_t5_select, input_attention_mask_select], dim=1)
                # inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                # encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)

                output_tokens_select = self.t5_tokenizer(
                    pseudo_label, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                targets_select = output_tokens_select.input_ids.masked_fill(
                    output_tokens_select.input_ids == self.t5_tokenizer.pad_token_id, -100)
                output_tokens_select_mask = output_tokens_select.attention_mask

                outputs_select = self.t5_model_select(
                    inputs_embeds=inputs_embeds_select, attention_mask=encoder_atts_select,
                    decoder_attention_mask=output_tokens_select_mask,
                    return_dict=True, labels=targets_select)
                loss_select = outputs_select.loss

            return {"loss": loss_select+loss_qa}

        elif 'select' not in self.task or 'train_qa_wo_selector' in self.task:

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            image_embeds = self.ln_vision(image_embeds)

            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            inputs_t5_qa = torch.repeat_interleave(inputs_t5_qa, n, 0)
            atts_t5 = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)
            if self.prompt.count("{}") == 2:
                text_input_qa = [self.prompt.format(question, passage) for question, passage in
                              zip(samples["text_input"], samples["passages"])]
            else:
                text_input_qa = [self.prompt.format(question) for question in samples["text_input"]]
            answer = samples["gold_answer"]

            with self.maybe_autocast(dtype=torch.bfloat16):
                input_tokens = self.t5_tokenizer(
                    text_input_qa,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                output_tokens = self.t5_tokenizer(
                    answer,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
                )

                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_t5_qa, inputs_embeds], dim=1)

                outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                )

                loss = outputs.loss

                return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=256,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        use_nucleus_sampling=False,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        batch_size = samples["image"].size(0)
        if "passages" in samples:
            if type(samples["passages"][0]) is list:
                n = len(samples["passages"][0])
                new_text_input = []
                new_passages = []
                for b, passage in enumerate(samples["passages"]):
                    new_passages.extend(passage)
                    new_text_input.extend([samples["text_input"][b]] * n)
                samples["passages"] = new_passages
                samples["text_input"] = new_text_input
        image = samples["image"]

        if 'select' not in self.task:

            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            if self.prompt.count("{}") == 2:
                text_input = [self.prompt.format(question, passage) for question, passage in
                              zip(samples["text_input"], samples["passages"])]
            else:
                text_input = [self.prompt.format(question) for question in samples["text_input"]]
            # doc_scores = torch.tensor([passage['score'] for passage in
            #                            samples["passages"]], dtype=torch.float).to(image.device)
            # doc_scores = doc_scores.reshape(batch_size, n)

            inputs_t5 = self.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            inputs_t5 = torch.repeat_interleave(inputs_t5, n, 0)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            input_tokens = self.t5_tokenizer(
                text_input, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            with self.maybe_autocast(dtype=torch.bfloat16):
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                    # return_dict_in_generate=True,
                    # output_scores=True,
                )

            # output_text = self.t5_tokenizer.batch_decode(
            #     outputs, skip_special_tokens=True
            # )
            # ans_per_ques = int(len(output_text) / batch_size)
            # new_question_id = []
            # for b in range(batch_size):
            #     new_question_id.extend([samples["question_id"][b]] * ans_per_ques)
            # samples["question_id"] = new_question_id

            #######采用不同输入的方式进行vote#########
            output_text = []
            generation_outputs_decoded = self.t5_tokenizer.batch_decode(outputs,
                                                                        skip_special_tokens=True)
            for b in range(batch_size):
                answer_proposals = generation_outputs_decoded[b * n:(b + 1) * n]
                output_text.append(max(answer_proposals, key=answer_proposals.count))

            # ########return the highest confident answer#########
            # generation_outputs = outputs['sequences']
            # generation_seq_scores = outputs['sequences_scores']
            # # print("generation_seq_scores", generation_seq_scores.shape)
            # generation_seq_scores = generation_seq_scores.reshape(batch_size, n)
            #
            # # reshape generation_outputs
            # generation_outputs = generation_outputs.reshape(batch_size, n, -1)
            #
            # # doc_scores_log --> log_softmax --> log(g(z))
            # # generation_seq_scores --> log(p(y|x, z))
            # # log(g(z)p(y|x, z)) = doc_scores + generation_seq_scores
            # # batch_size x n_docs + batch_size x n_docs
            # doc_scores_log = F.log_softmax(doc_scores, dim=-1)
            # # print('doc_scores_log', doc_scores_log)
            # # print('generation_seq_scores', generation_seq_scores)
            # loss_with_doc_scores = doc_scores_log + generation_seq_scores
            #
            # output_text = []
            # for b in range(batch_size):
            #     # use topk to get indices of top candidates
            #     top_cand_inds = (loss_with_doc_scores[b]).topk(1)[1]
            #     output_text.append(generation_outputs[b, top_cand_inds])
            # output_text = torch.cat(output_text)
            # output_text = self.t5_tokenizer.batch_decode(output_text, skip_special_tokens=True)
        # inference with selector
        else:
            with self.maybe_autocast():
                image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # b 256 c
            image_embeds_, image_atts_ = image_embeds.detach().clone(), image_atts.detach().clone()
            image_embeds_ = self.ln_vision_select(image_embeds_)

            text_input_select = [self.prompt.format(question, passage) for question, passage in
                                 zip(samples["text_input"], samples["passages"])]
            query_tokens_select = self.query_tokens_select.expand(image_embeds_.shape[0], -1, -1)
            query_output_select = self.Qformer_select.bert(
                query_embeds=query_tokens_select, encoder_hidden_states=image_embeds_,
                encoder_attention_mask=image_atts_, return_dict=True)
            inputs_t5_select = self.t5_proj_select(query_output_select.last_hidden_state)
            inputs_t5_select = torch.repeat_interleave(inputs_t5_select, n, 0)

            atts_t5_select = torch.ones(inputs_t5_select.size()[:-1], dtype=torch.long).to(image.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # frame_prefix = self.t5_tokenizer(
                #     self.frame_prefix, padding="longest", add_special_tokens=False,
                #     truncation=True, max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                # frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, b * t, 0)
                # frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, b * t, 0)
                # frame_predix_embed = self.t5_model.encoder.embed_tokens(frame_prefix_id)

                input_tokens_select = self.t5_tokenizer(
                    text_input_select, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                input_ids_select = input_tokens_select.input_ids
                input_attention_mask_select = input_tokens_select.attention_mask
                # input_ids_select = torch.repeat_interleave(input_tokens_select.input_ids, t, 0)
                # input_attention_mask_loc = torch.repeat_interleave(input_tokens_select.attention_mask, t, 0)
                inputs_embeds_select = self.t5_model_select.encoder.embed_tokens(input_ids_select)
                inputs_embeds_select = torch.cat([inputs_t5_select, inputs_embeds_select], dim=1)
                encoder_atts_select = torch.cat([atts_t5_select, input_attention_mask_select], dim=1)
                # inputs_embeds_loc = torch.cat([frame_predix_embed, inputs_t5_loc, inputs_embeds_loc], dim=1)
                # encoder_atts_loc = torch.cat([frame_prefix_mask, atts_t5_loc, input_attention_mask_loc], dim=1)

                outputs_select = self.t5_model_select.generate(
                    inputs_embeds=inputs_embeds_select, attention_mask=encoder_atts_select,
                    do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature, num_beams=1,
                    max_new_tokens=max_len, min_length=min_len, repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty, num_return_sequences=num_captions,
                    return_dict_in_generate=True, output_hidden_states=True, output_scores=True)

                pred_logits_select = outputs_select.scores[0]
                select_yes = pred_logits_select[:, self.yes_id]
                select_yes = select_yes.reshape(batch_size, -1)

                # generation_outputs = outputs_select['sequences']
                # generation_texts = self.t5_tokenizer.batch_decode(generation_outputs,
                #                                                                skip_special_tokens=True)
                # print(generation_texts)

            select_knowledge_idx = torch.topk(select_yes, self.knowledge_num, dim=-1).indices.tolist()
            sorted_knowledge_idx = []
            for k in select_knowledge_idx:
                sorted_knowledge_idx.append(sorted(k))

            # sorted_knowledge_idx = [random.sample(range(30), self.knowledge_num) for j in range(batch_size)]

            ### 分别输入 ###
            text_input_qa = [self.prompt.format(question, passage) for question, passage in
                             zip(samples["text_input"], samples["passages"])]
            select_text_input_qa = []
            for i, ks in enumerate(sorted_knowledge_idx):
                for j, k in enumerate(ks):
                    select_text_input_qa.append(text_input_qa[i * n + k])

            ### concat ###
            # select_question = []
            # select_passage = []
            # for i, ks in enumerate(sorted_knowledge_idx):
            #     docs = []
            #     for j, k in enumerate(ks):
            #         docs.append(samples["passages"][i * n + k])
            #     select_question.append(samples["text_input"][i * n])
            #     select_passage.append(' '.join(docs))
            # select_text_input_qa = [self.prompt.format(question, passage) for question, passage in
            #                         zip(select_question, select_passage)]

            image_embeds = self.ln_vision(image_embeds)  # b, 256,c
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # bt n c
            query_tokens_qa = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output_qa = self.Qformer.bert(
                query_embeds=query_tokens_qa, encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5_qa = self.t5_proj(query_output_qa.last_hidden_state)
            inputs_t5_qa = torch.repeat_interleave(inputs_t5_qa, self.knowledge_num, 0)
            atts_t5_qa = torch.ones(inputs_t5_qa.size()[:-1], dtype=torch.long).to(image.device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # vid_prefix = self.t5_tokenizer(
                #     self.vid_prefix, padding="longest", add_special_tokens=False,
                #     truncation=True, max_length=self.max_txt_len, return_tensors="pt", ).to(image.device)  #
                # vid_prefix_id = torch.repeat_interleave(vid_prefix.input_ids.unsqueeze(0), b, 0)
                # vid_prefix_mask = torch.repeat_interleave(vid_prefix.attention_mask.unsqueeze(0), b, 0)
                # vid_prefix_embed = self.t5_model.encoder.embed_tokens(vid_prefix_id)  # b t n_word c

                # inputs_t5_qa = torch.cat([vid_prefix_embed, inputs_t5_qa], dim=2)  # b, t, n_word + m, c
                # atts_t5_qa = torch.cat([vid_prefix_mask, atts_t5_qa], dim=2)  # b, t, n_word + m
                # inputs_t5_qa = inputs_t5_qa.reshape(b, -1, inputs_t5_qa.shape[-1])
                # atts_t5_qa = atts_t5_qa.reshape(b, -1)

                input_tokens_qa = self.t5_tokenizer(
                    select_text_input_qa, padding="longest", truncation=True,
                    max_length=self.max_txt_len, return_tensors="pt").to(image.device)
                inputs_embeds_qa = self.t5_model.encoder.embed_tokens(input_tokens_qa.input_ids)
                inputs_embeds_qa = torch.cat([inputs_t5_qa, inputs_embeds_qa], dim=1)
                encoder_atts_qa = torch.cat([atts_t5_qa, input_tokens_qa.attention_mask], dim=1)

                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds_qa,
                    attention_mask=encoder_atts_qa,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                )

            # output_text = self.t5_tokenizer.batch_decode(
            #     outputs, skip_special_tokens=True
            # )
            # ans_per_ques = int(len(output_text) / batch_size)
            # new_question_id = []
            # for b in range(batch_size):
            #     new_question_id.extend([samples["question_id"][b]] * ans_per_ques)
            # samples["question_id"] = new_question_id
            #######采用不同输入的方式进行vote#########
            output_text = []
            generation_outputs_decoded = self.t5_tokenizer.batch_decode(outputs,
                                                                               skip_special_tokens=True)
            for b in range(batch_size):
                answer_proposals = generation_outputs_decoded[b * self.knowledge_num:(b + 1) * self.knowledge_num]
                output_text.append(max(answer_proposals,key=answer_proposals.count))

        return output_text
        # return sorted_knowledge_idx

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        use_lora = cfg.get("use_lora", True)
        qformer_text_input = cfg.get("qformer_text_input", True)
        task = cfg.get("task", 'train_qa')
        knowledge_num = cfg.get("knowledge_num", 5)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            task=task,
            knowledge_num=knowledge_num,
            use_lora=use_lora,
        )

        if cfg.load_finetuned == True:
            cfg.load_finetuned = False
            cfg.load_pretrained = True
            model.load_checkpoint_from_config(cfg)  ##保存模型参数的时候没有保存q-former的参数，推理时需要额外加载
            cfg.load_finetuned = True
            cfg.load_pretrained = False
            model.load_checkpoint_from_config(cfg)
        else:
            model.load_checkpoint_from_config(cfg)
            if 'select' in task:
                model.t5_proj_select.load_state_dict(model.t5_proj.state_dict())
                # model.t5_proj_select = copy.deepcopy(model.t5_proj)

        ##加载answerer或者selector的linear和lora参数
        # logging.info("load linear and lora")
        # url_or_filename = "/public/dzhao/blip2_vqa/lavis/output/blip2_flant5xl/OKVQA_select_lora/train_cycle_select_bs8_lr1e-4_k30_5/sota/checkpoint_best.pth"
        # checkpoint = torch.load(url_or_filename, map_location="cpu")
        # state_dict = checkpoint["model"]
        # msg = model.load_state_dict(state_dict, strict=False)
        # logging.info("Missing keys {}".format(msg.missing_keys))
        # logging.info("load checkpoint from %s" % url_or_filename)

        return model
