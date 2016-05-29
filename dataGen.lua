
local Data = torch.class 'Data'

function Data:__init()
  dataset = dofile 'dataset.lua'
  dataset.download_generate_bin()
  self.trainData = dataset.get_train_dataset()
  self.testData = dataset.get_test_dataset()

  self:normalize_global()

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
