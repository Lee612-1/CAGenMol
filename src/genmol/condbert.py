import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForMaskedLM, MaskedLMOutput


class CondBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        hidden = config.hidden_size
        # Shared bias that is added to every conditional token embedding
        self.cond_bias = nn.Parameter(torch.zeros(hidden))

    @torch.no_grad()
    def _word_embeds(self, input_ids):
        # Convenience wrapper around the embedding layer
        return self.bert.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cond_vec=None,     # cond_vec can be [B, H] or [B, X, H]
    ):
        # ------------------------------------------------------------------
        # 1) Build token embeddings for the original sentence (B, L, H)
        #    and prepare conditional token embeddings (B, cond_len, H)
        # ------------------------------------------------------------------
        if inputs_embeds is not None:
            raise ValueError("Please pass input_ids only; this module builds inputs_embeds internally.")
        if input_ids is None:
            raise ValueError("input_ids is required.")

        B, L = input_ids.shape
        device = input_ids.device
        H = self.config.hidden_size

        word_embeds = self._word_embeds(input_ids)  # (B, L, H)

        # ---- handle different shapes of cond_vec ----
        if cond_vec is None:
            # No condition provided: use a single learned bias token
            cond_len = 1
            cond_tokens = self.cond_bias.view(1, 1, H).expand(B, 1, H)  # (B, 1, H)
        else:
            if cond_vec.dim() == 2:
                # (B, H) -> treat as a single conditional token
                cond_len = 1
                cond_tokens = cond_vec.unsqueeze(1)  # (B, 1, H)
            elif cond_vec.dim() == 3:
                # (B, X, H)
                if cond_vec.size(0) != B or cond_vec.size(-1) != H:
                    raise ValueError(
                        f"cond_vec shape mismatch: expected (B, X, H) with B={B}, H={H}, got {cond_vec.shape}"
                    )
                cond_len = cond_vec.size(1)
                cond_tokens = cond_vec  # (B, X, H)
            else:
                raise ValueError(f"Unsupported cond_vec ndim={cond_vec.dim()}, expected 2 or 3.")

            # Add a shared bias to all conditional tokens
            cond_tokens = cond_tokens + self.cond_bias.view(1, 1, H)

        # ------------------------------------------------------------------
        # 2) Prepend conditional tokens to the sequence
        # ------------------------------------------------------------------
        # cond_tokens: (B, cond_len, H), word_embeds: (B, L, H)
        inputs_embeds = torch.cat([cond_tokens, word_embeds], dim=1)  # (B, cond_len + L, H)

        # ------------------------------------------------------------------
        # 3) Extend attention_mask / token_type_ids / position_ids if provided
        # ------------------------------------------------------------------
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=torch.long)

        # Condition tokens are always "visible"
        cond_attn = torch.ones(B, cond_len, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([cond_attn, attention_mask], dim=1)  # (B, cond_len + L)

        if token_type_ids is not None:
            # Put condition tokens into segment 0
            cond_token_type = torch.zeros(B, cond_len, device=device, dtype=token_type_ids.dtype)
            token_type_ids = torch.cat([cond_token_type, token_type_ids], dim=1)

        if position_ids is not None:
            # Original position_ids corresponds to (B, L). After prepending cond_len tokens:
            #   - condition positions: 0..cond_len-1
            #   - original positions: shifted by +cond_len
            if position_ids.size(1) != L:
                raise ValueError(f"position_ids length mismatch: {position_ids.size(1)} vs input length {L}")

            cond_pos = torch.arange(cond_len, device=device, dtype=position_ids.dtype).unsqueeze(0).expand(B, cond_len)
            position_ids = torch.cat([cond_pos, position_ids + cond_len], dim=1)  # (B, cond_len + L)

        # ------------------------------------------------------------------
        # 4) Run the BERT encoder using inputs_embeds
        # ------------------------------------------------------------------
        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
        )

        # ------------------------------------------------------------------
        # 5) Drop the conditional prefix and run the MLM head
        # ------------------------------------------------------------------
        # Slice away the conditional tokens so logits align with labels (B, L)
        sequence_output = outputs.last_hidden_state[:, cond_len:, :]  # (B, L, H)
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # labels is still (B, L); no shifting needed
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            out = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + out) if masked_lm_loss is not None else out

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


