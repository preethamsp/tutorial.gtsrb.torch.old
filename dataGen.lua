require 'image'
require 'nn'


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

function Data:normalize()
    local trainData = self.trainData
    local testData = self.testData

    print '<trainer> preprocessing data (color space + normalization)'
    collectgarbage()

    -- preprocess trainSet
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    for i = 1,trainData.data:size(1) do
       xlua.progress(i, trainData.data:size(1))
       -- rgb -> yuv
       local rgb = trainData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[1] = normalization(yuv[{{1}}])
       trainData.data[i] = yuv
    end
    -- normalize u globally:
    local mean_u = trainData.data:select(2,2):mean()
    local std_u = trainData.data:select(2,2):std()
    trainData.data:select(2,2):add(-mean_u)
    trainData.data:select(2,2):div(std_u)
    -- normalize v globally:
    local mean_v = trainData.data:select(2,3):mean()
    local std_v = trainData.data:select(2,3):std()
    trainData.data:select(2,3):add(-mean_v)
    trainData.data:select(2,3):div(std_v)

    trainData.mean_u = mean_u
    trainData.std_u = std_u
    trainData.mean_v = mean_v
    trainData.std_v = std_v

    -- preprocess testSet
    for i = 1,testData.data:size(1) do
      xlua.progress(i, testData.data:size(1))
       -- rgb -> yuv
       local rgb = testData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[{1}] = normalization(yuv[{{1}}])
       testData.data[i] = yuv
    end
    -- normalize u globally:
    testData.data:select(2,2):add(-mean_u)
    testData.data:select(2,2):div(std_u)
    -- normalize v globally:
    testData.data:select(2,3):add(-mean_v)
    testData.data:select(2,3):div(std_v)
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
