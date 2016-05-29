dofile 'dataGen.lua'
data = Data()
data:normalize()
torch.save('/fast_data/gtsrb.tutorial/data.t7',data)
