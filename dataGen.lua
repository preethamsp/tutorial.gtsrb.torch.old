
local Data = torch.class 'Data'

function Data:__init()
  dataset = dofile 'dataset.lua'
  dataset.download_generate_bin()
  self.trainData = dataset.get_train_dataset()
  self.testData = dataset.get_test_dataset()

  self:normalize_local(self.trainData)
  self:normalize_local(self.testData)

end

function Data:TrainGenerator(batchSize)
  batchSize = batchSize or 32
  local indices = torch.randperm(self.trainData.data:size(1)):long():split(batchSize)
  local idx = 0
  return function()
    idx = idx + 1
    if idx <= #indices then
      local X = self.trainData.data:index(1,indices[idx])
      local Y = self.trainData.label:index(1,indices[idx])
      return X,Y
    end
  end
end

function Data:TestGenerator(batchSize)
  batchSize = batchSize or 32
  local indices = torch.range(1,self.testData.data:size(1)):long():split(batchSize)
  local idx = 0
  return function()
    idx = idx + 1
    if idx <= #indices then
      local X = self.testData.data:index(1,indices[idx])
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

function Data:normalize_local(dataset)
  require 'image'
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  local batch = 200 -- Can be reduced if you experience memory issues
  local dataset_size = dataset.data:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset.data:narrow(1,i,local_batch))
    dataset.data:narrow(1,i,local_batch):copy(normalized_images)
  end
end


-- normalize the dataset (supress the mean and divide by standard deviation)
function Data:normalize_global()
  local std = self.trainData.data:std()
  local mean = self.trainData.data:mean()
  self.trainData.data:add(-mean)
  self.trainData.data:div(std)
  self.testData.data:add(-mean)
  self.testData.data:div(std)
  return mean, std
end
