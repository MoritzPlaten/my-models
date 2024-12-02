from Layers.BaseAttention import BaseAttention

class CausalSelfAttention(BaseAttention):

  def call(self, x, training=False):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x