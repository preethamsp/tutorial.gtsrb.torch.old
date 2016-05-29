require 'image'
require 'nn'
local t = require 'transforms.lua'

local Data = torch.class 'Data'

function Data:__init()
  dataset = dofile 'dataset.lua'
  dataset.download_generate_bin()
  self.trainData = dataset.get_train_dataset()
  self.testData = dataset.get_test_dataset()

end

function Data:TrainGenerator(batchSize)
  batchSize = batchSize or 32
  local indices = torch.randperm(self.trainData.data:size(1)):long():split(batchSize)
  local idx = 0
  local trainPreprocess = t.Compose{
      t.ColorJitter({
          brightness = 0.4,
          contrast = 0.4,
          saturation = 0.4,
      }),
      t.Lighting(0.1, t.pca.eigval, t.pca.eigvec),
      t.ColorNormalize(t.meanstd)}

  return function()
    idx = idx + 1
    if idx <= #indices then
      local X = self.trainData.data:index(1,indices[idx])
      for i = 1,X:size(1) do
        X[i] = trainPreprocess(X[i])
      end
      local Y = self.trainData.label:index(1,indices[idx])
      return X,Y
    end
  end
end

function Data:TestGenerator(batchSize)
  batchSize = batchSize or 32
  local indices = torch.range(1,self.testData.data:size(1)):long():split(batchSize)
  local idx = 0
  local testPreprocess = t.Compose{
       t.ColorNormalize(t.meanstd),
     }
  return function()
    idx = idx + 1
    if idx <= #indices then
      local X = self.testData.data:index(1,indices[idx])
      for i = 1,X:size(1) do
        X[i] = testPreprocess(X[i])
      end
      local Y = self.testData.label:index(1,indices[idx])
      return X,Y
    end
  end
end

function Data:getTrainDataSize()
  return self.trainData.data:size(1)
end

function Data:getTestDataSize()
  return self.testData.data:size(1)
end
