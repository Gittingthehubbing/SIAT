from torch import nn
import torch as t
import transformers
import einops as eo


class RegressionLayers(nn.Module):

    def __init__(self,embed_dim,num_targets,use_first_embedding=False,keep_dimension=False,use_layer_norm=True, num_lin_layers=1,dropout_lin_layer = 0) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.use_first_embedding = use_first_embedding
        self.keep_dimension = keep_dimension
        self.use_layer_norm = use_layer_norm
        self.num_lin_layers = num_lin_layers
        self.dropout_lin_layer = dropout_lin_layer

        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(embed_dim))
        for _ in range(num_lin_layers-1):
            layers.append(nn.Linear(embed_dim,embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_lin_layer))

        layers.append(nn.Linear(embed_dim,num_targets))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        if not self.keep_dimension:
            if self.use_first_embedding:

                if x.ndim >3:
                    x = x[:,:,0,:]
                else:
                    x = x[:,0]
            else:

                if x.ndim >3:
                    x = eo.reduce(x,'b s n e -> b s e', reduction='mean') 
                else:
                    x = eo.reduce(x,'b n e -> b e', reduction='mean') 

        return self.model(x)


class SIAT(nn.Module):
    def __init__(
        self,
        x_shape_enc,
        num_targets,
        depth_dec,
        heads_dec,
        dropout_dec,
        backbone,
        gpt2_inter_dim,
        use_first_embedding=True,
        loop_over_timesteps=True,
        num_lin_layers=1,
        dropout_lin_layer = 0,
        num_future_steps_to_predict = 3,
    ):

        super().__init__()
        self.modelName = "SIAT"
        self.x_shape_enc = x_shape_enc
        self.num_targets = num_targets
        self.depth_dec = depth_dec
        self.heads_dec = heads_dec
        self.use_first_embedding = use_first_embedding
        self.loop_over_timesteps = loop_over_timesteps
        self.num_lin_layers = num_lin_layers
        self.dropout_lin_layer = dropout_lin_layer
        self.dropout_dec = dropout_dec

        if loop_over_timesteps:
            self.num_future_steps_to_predict = num_future_steps_to_predict
        else:
            self.num_future_steps_to_predict = 1
        self.embed_and_encode = backbone
        with t.no_grad():
            self.x_shape_dec = self.get_decoder_input_shape(t.randn(x_shape_enc))
        self.embed_dim_dec = self.x_shape_dec[-1] 
        
        self.gpt2_inter_dim = gpt2_inter_dim

        self.gpt_vocab = self.embed_dim_dec

        self.gpt_encoder = nn.Linear(self.embed_dim_dec,gpt2_inter_dim,bias=False)
        gpt_config = transformers.GPT2Config(n_embd=gpt2_inter_dim,
                                            vocab_size=self.gpt_vocab,
                                            use_cache=True,
                                            n_head=heads_dec,
                                            n_layer=depth_dec,
                                            resid_pdrop=self.dropout_dec,
                                            embd_pdrop=self.dropout_dec,
                                            attn_pdrop=self.dropout_dec,
                                            n_positions =self.x_shape_dec[0],
                                            )
        self.gpt_model = transformers.GPT2Model(gpt_config)

        self.gpt_decoder = nn.Linear(gpt2_inter_dim,self.embed_dim_dec,bias=False)
       
        self.dropout_dec = dropout_dec
        self.regression = RegressionLayers(
            self.embed_dim_dec,num_targets,
            use_first_embedding=use_first_embedding,
            keep_dimension=True,
            use_layer_norm=False,
            num_lin_layers=num_lin_layers,
            dropout_lin_layer=dropout_lin_layer
        )

    def forward(self,x):
        """Input shape [b,t,c,h,w]
        Example for sequence of 8 images with batch size 16 [16,8,3,224,224]"""
        x_encoded, x_predicted_future_encoding, regression_out = self.forward_partial(x)

        if regression_out.ndim>2 and regression_out.shape[-1] == 1:
            regression_out = regression_out.squeeze(-1)
        return x_encoded, x_predicted_future_encoding, regression_out

    def forward_partial(self,x):
        needs_unfloding = False
        if x.ndim >4: 
            needs_unfloding = True
            batch_size = x.shape[0]
            x = eo.rearrange(x, 'b t c h w -> (b t) c h w')
        
        x_encoded = self.embed_and_encode(x)
        
        if needs_unfloding:
            x_encoded = eo.rearrange(x_encoded, '(b t) e -> b t e' ,b=batch_size) 
        x_encoded_gpt_in = self.gpt_encoder(x_encoded[:,:-self.num_future_steps_to_predict])
        if self.loop_over_timesteps:
            past = None
            all_outputs = []
            all_outputs_decoded = []
            for out_id in range(self.num_future_steps_to_predict):
                predictions_so_far = sum([el.size(1) for el in all_outputs])
                position_ids = t.arange(predictions_so_far,predictions_so_far + x_encoded_gpt_in.size(1),dtype=t.long,device=x_encoded_gpt_in.device)

                gpt_out = self.gpt_model(
                    inputs_embeds=x_encoded_gpt_in,
                    past_key_values = past,
                    position_ids = position_ids
                )
                last_hidden_state = gpt_out.last_hidden_state
                all_outputs.append(last_hidden_state)
                all_outputs_decoded.append(self.gpt_decoder(last_hidden_state))
                past = gpt_out.past_key_values
                x_encoded_gpt_in = last_hidden_state[:, -1:, :]
            x_predicted_future_encoding = t.cat(all_outputs_decoded,dim=1)

        else:
            position_ids = t.arange(0,x_encoded_gpt_in.size(1),dtype=t.long,device=x_encoded.device)
            
            gpt_out = self.gpt_model(
                inputs_embeds=x_encoded_gpt_in,
                past_key_values = None,
                position_ids = position_ids
            )

            last_hidden_state = gpt_out.last_hidden_state
            past = gpt_out.past_key_values
            x_predicted_future_encoding = self.gpt_decoder(last_hidden_state)
            
        regression_in = t.cat([x_encoded[:,:1,:],x_predicted_future_encoding],dim=1)
        regression_out = self.regression(regression_in).squeeze(-1)
        return x_encoded, x_predicted_future_encoding, regression_out

    def get_decoder_input_shape(self,x):     
        if x.ndim <4:
            x= x.unsqueeze(0)     
        
        x_encoded= self.embed_and_encode(x)
        
        if self.use_first_embedding and x_encoded.ndim >2:
            x_encoded = x_encoded[:,0,:]
        elif x_encoded.ndim >2:
            x_encoded = eo.reduce(x_encoded,'b n e -> b e',reduction="mean") 
                
        return x_encoded.shape