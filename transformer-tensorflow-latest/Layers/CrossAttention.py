from Layers.BaseAttention import BaseAttention

class CrossAttention(BaseAttention):

  def call(self, x, context, training=False):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True,
        training=training)

    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x