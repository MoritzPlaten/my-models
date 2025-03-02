from Layers.BaseAttention import BaseAttention

class GlobalSelfAttention(BaseAttention):

  def call(self, x, training=False): #TODO: add mask=None
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        training=training)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
