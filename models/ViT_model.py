from torch import nn

class PretrainedTimmNet(nn.Module):
    
    def __init__(
        self,
        inmodel,
        numberOfLinearLayers,
        dropOutLin,
        intermediateLinearLayerShape,
        linearActivationFunc,
        sigmoidOn,
        yShape,
        encoding_only=False,
        use_both_exposures_SIRTA=False
    ) -> None:
        super().__init__()

        self.modelName = "pretrained_timm_net"
        self.numberOfLinearLayers = numberOfLinearLayers
        self.dropOutLin = dropOutLin
        self.intermediateLinearLayerShape = intermediateLinearLayerShape
        self.linearActivationFunc = linearActivationFunc
        self.sigmoidOn = sigmoidOn
        self.linInShape = inmodel.norm.normalized_shape[0]
        self.encoding_only = encoding_only
        self.use_both_exposures_SIRTA = use_both_exposures_SIRTA

        if len(yShape)>1:
            self.outShape = yShape[1]
        else:
            self.outShape = yShape[0]

        if self.use_both_exposures_SIRTA:
            self.im_project = nn.Conv2d(6,3,3,1,1)
        else:
            self.im_project = nn.Identity()

        self.vit_main = inmodel    

        if encoding_only:
            self.linModel = None
        else:
            linearLayers = []
            layerInShape = self.linInShape
            for linLayer in range(1,numberOfLinearLayers):
                linearLayers.append(nn.Dropout(dropOutLin))
                linearLayers.append(nn.Linear(layerInShape,intermediateLinearLayerShape))
                layerInShape = intermediateLinearLayerShape
                if linLayer < numberOfLinearLayers:
                    if linearActivationFunc != "Linear":
                        linearLayers.append(getattr(nn,linearActivationFunc)())
            linearLayers.append(nn.Dropout(dropOutLin))
            linearLayers.append(nn.Linear(layerInShape,self.outShape))

            if sigmoidOn:
                linearLayers.append(nn.Sigmoid())

            self.linModel = nn.Sequential(*linearLayers)
    
    def forward(self,x):
        x = self.im_project(x)
        x = self.vit_main(x)

        if self.encoding_only:
            return x
        else:
            linOut = self.linModel(x)
            return linOut
    
    def remove_final_linear(self):
        self.linModel = nn.Identity()
