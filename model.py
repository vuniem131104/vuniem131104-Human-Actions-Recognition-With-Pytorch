import torch
import torch.nn as nn

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False):
        if not isinstance(module, nn.Module):
            raise ValueError(
                "Please initialize `TimeDistributed` with a "
                f"`torch.nn.Module` instance. Received: {module.type}"
            )
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if self.batch_first:
            # BS, Seq_length, * -> Seq_length, BS, *
            x = x.transpose(0, 1)
        output = torch.stack([self.module(xt) for xt in x], dim=0)
        if self.batch_first:
            # Seq_length, BS, * -> BS, Seq_length, *
            output = output.transpose(0, 1)
        return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3), (1,1,1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((1,3,3), (1,2,2)),
            nn.Dropout(0.3),
            nn.Conv3d(8, 16, (1, 3, 3), (1,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1,3,3), (1,2,2)),
            nn.Dropout(0.3),
            nn.Conv3d(16, 32, (1, 3, 3), (1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,3,3), (1,2,2)),
            nn.Dropout(0.3),
            nn.Conv3d(32, 16, (1, 3, 3), (1,1,1))
        )
        self.linear = nn.Linear(2880, 4)

    def forward(self, x):
        x = self.model(x)
        a = x.size(0)
        x = x.view(a, -1)
        return self.linear(x)

class Model2(nn.Module):
  def __init__(self):
      super().__init__()
      self.model = nn.Sequential(
          TimeDistributed(nn.Conv2d(3, 16, (3, 3), padding='same')),
          nn.ReLU(),
          TimeDistributed(nn.MaxPool2d((3, 3), stride=4)),
          nn.Dropout(0.25),
          TimeDistributed(nn.Conv2d(16, 32, (3, 3), padding='same')),
          nn.ReLU(),
          TimeDistributed(nn.MaxPool2d((3, 3), stride=4)),
          nn.Dropout(0.25),
          TimeDistributed(nn.Conv2d(32, 64, (3, 3), padding='same')),
          nn.ReLU(),
          TimeDistributed(nn.MaxPool2d((2, 2), stride=2)),
          nn.Dropout(0.25),
          TimeDistributed(nn.Conv2d(64, 64, (3, 3), padding='same')),
          nn.ReLU(),
          TimeDistributed(nn.MaxPool2d((2, 2), stride=2))
      )
      self.linear = nn.Linear(1280, 4)

  def forward(self, x):
      x = self.model(x)
      a = x.size(0)
      x = x.reshape(a, -1)
      return self.linear(x)