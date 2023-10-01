import sys
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast


# -------------------------for pos neg training------------------------
def compute_contrastive_loss(sample_type, logits, labels, vocab_size):
    """
    expecting number of elements to be divisbile by 3
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    #
    loss = 0.0
    count_invals = 0
    for l in range(shift_labels.shape[0]):
        # 2,4 signifies pos and neutral sample
        if sample_type[l] in [2, 4]:
            loss_fct = CrossEntropyLoss()
            nl = loss_fct(shift_logits[l, :, :], shift_labels[l, :])
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                # print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl = torch.tensor(0.0).to(shift_labels.device)
                count_invals += 1

            loss += nl

        elif sample_type[l] == 3:
            # 3 signifies neg sample
            loss_fct = CrossEntropyLoss()
            nl_neg = loss_fct(shift_logits[l, :, :], shift_labels[l, :])
            if nl_neg > 1:
                nl = nl_neg
                print("--------------neg sample loss is minimized----------------")
            else:
                nl = -0.1 * nl_neg
            print(f"#={l}:type={sample_type[l]}:loss={nl}")
            if torch.isnan(nl):
                # print(f"Nan loss for: {sample_type[l]} \n{shift_labels[l,:][100:500]}")
                nl_neg = torch.tensor(0.0).to(shift_labels.device)
                nl = nl_neg
                count_invals += 1

            loss += nl
        else:
            breakpoint()
            print(">>>>Error: Not sure if it is positive sample or negative")
            sys.exit()

    print(f"count NANs: {count_invals}")
    print(f"div factor: {shift_labels.shape[0]-count_invals}")
    loss = loss / (shift_labels.shape[0] - count_invals)
    print(f"\t\t\tOverall loss={loss}")
    return loss


def compute_loss(logits, labels):
    """
    expecting number of elements to be divisbile by 3
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    count_invals = 0
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    if torch.isnan(loss):
        breakpoint()
        loss = torch.tensor(0.0).to(shift_labels.device)
        count_invals += 1

    print(f"count NANs: {count_invals}")
    print(f"div factor: {shift_labels.shape[0]-count_invals}")
    loss = loss / (shift_labels.shape[0] - count_invals)
    print(f"\t\t\tOverall loss={loss}")
    return loss


def nested_detach(obj):
    """
    Recursively detach tensors in an object.

    :param obj: The object which may contain tensor(s) to detach.
    :type obj: torch.Tensor or dict
    :return: The object with detached tensor(s).
    :rtype: type(obj)
    """
    if isinstance(obj, int): return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: nested_detach(v) for k, v in obj.items()}
    else:
        breakpoint()
        raise TypeError("Object must be a tensor or a (nested) dictionary of tensors.")


class CustomTrainer(Trainer):
    # modified by emrys
    def _compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # correcting the first token
        sample_type = torch.clone(inputs["input_ids"][:, 0])
        inputs["input_ids"][:, 0] = 1
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # loss = outputs["loss"]['sum_loss']
        loss = compute_contrastive_loss(
            sample_type,
            outputs["logits"],
            outputs["loss"]["labels"],
            outputs["loss"]["vocab_size"],
        )
        # print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss

    # added by emrys
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # loss = outputs["loss"]['sum_loss']
        mid = len(outputs["logits"]) // 2
        positive_loss = compute_loss(
            outputs["logits"][:mid],
            outputs["loss"]["labels"][:mid],
        )
        negative_loss = compute_loss(
            outputs["logits"][mid:], outputs["loss"]["labels"][mid:]
        )
        loss = positive_loss - 0.1 * negative_loss
        # print('Loss:',loss)
        return (loss, outputs) if return_outputs else loss

    # LLaMA custom loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # device = model.device
        # for k in inputs:
        #     inputs[k] = inputs[k].to(device)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            mid = len(outputs["logits"]) // 2
            positive_loss = compute_loss(
                outputs["logits"][:mid],
                outputs["loss"]["labels"][:mid],
            )
            negative_loss = compute_loss(
                outputs["logits"][mid:], outputs["loss"]["labels"][mid:]
            )
            loss = positive_loss - 0.1 * negative_loss
            loss, positive_loss, negative_loss, outputs = (
                nested_detach(loss),
                nested_detach(positive_loss),
                nested_detach(negative_loss),
                nested_detach(outputs)
            )

        return loss, outputs, outputs["loss"]["labels"]


class CustomLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss={
                "sum_loss": loss,
                "logits": logits,
                "labels": labels,
                "vocab_size": self.config.vocab_size,
            },
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
