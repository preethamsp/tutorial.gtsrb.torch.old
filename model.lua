require 'nn'
nbClasses = 43
net = nn.Sequential()
--[[building block, adds a convolution layer, batch norm layer and a relu activation to the net]]--
function ConvBNReLU(nInputPlane, nOutputPlane)
-- kernel size = (3,3), stride = (1,1), padding = (1,1)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  net:add(nn.ReLU(true))
end

ConvBNReLU(3,32)
ConvBNReLU(32,32)
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.Dropout(0.3))
ConvBNReLU(32,64)
ConvBNReLU(64,64)
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.Dropout(0.3))
ConvBNReLU(64,128)
ConvBNReLU(128,128)
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.Dropout(0.3))
net:add(nn.View(128*6*6))
net:add(nn.Linear(128*6*6,512))
net:add(nn.Linear(512,nbClasses))

return net
