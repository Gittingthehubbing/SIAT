from models.ViT_model import PretrainedTimmNet
from models.SIAT_model import SIAT
import timm
import torch as t


nNet = timm.create_model(
    "deit3_medium_patch16_224.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)

backbone = PretrainedTimmNet(
    nNet,
    yShape=(1,),
    numberOfLinearLayers=1,
    dropOutLin=0.1,
    intermediateLinearLayerShape=None,
    sigmoidOn = False,
    linearActivationFunc=None,
    encoding_only = True,
    use_both_exposures_SIRTA=False
)

xin = t.rand(2,3,224,224)

with t.inference_mode():
    out = backbone(xin)

print(f"Backbone Model has output shape {out.shape}")

propDict = dict(
    gpt2_inter_dim_multi_factor = 4,
    x_shape_enc = (8,3,224,224), # (seqlen,c,h,w)
    num_targets = 1,
    depth_dec = 4,
    heads_dec = 4,
    dropout_dec = 0.1,
    backbone = backbone,
    use_first_embedding = True,
    loop_over_timesteps = True,
    num_lin_layers = 1,
    dropout_lin_layer = 0.1,
    num_future_steps_to_predict = 3,
)
propDict["gpt2_inter_dim"] = propDict["gpt2_inter_dim_multi_factor"] * propDict["heads_dec"]

siat = SIAT(
        x_shape_enc = propDict["x_shape_enc"],
        num_targets = propDict["num_targets"],
        depth_dec = propDict["depth_dec"],
        heads_dec = propDict["heads_dec"],
        dropout_dec = propDict["dropout_dec"],
        backbone = propDict["backbone"],
        gpt2_inter_dim = propDict["gpt2_inter_dim"],
        use_first_embedding = propDict["use_first_embedding"],
        loop_over_timesteps = propDict["loop_over_timesteps"],
        num_lin_layers = propDict["num_lin_layers"],
        dropout_lin_layer = propDict["dropout_lin_layer"],
        num_future_steps_to_predict = propDict["num_future_steps_to_predict"],
)

x_seq_in = t.rand(2,8,3,224,224)

with t.inference_mode():
    seq_out = siat(x_seq_in)

print(f"SIAT Model has output shapes {[x.shape for x in seq_out]}")

print(f"""Irradiance predictions would be
 {seq_out[-1][:,:-propDict['num_future_steps_to_predict']].numpy()}""")
