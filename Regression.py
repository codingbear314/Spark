import torch
import torch.nn as nn

class Finance_001_Model(nn.Module):
    def __init__(self):
        super(Finance_001_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8)
        conv1_out_side = 60 - 8 + 1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8)
        conv2_out_side = conv1_out_side - 8 + 1
        self.dense1 = nn.Linear(32 * conv2_out_side, 40)
        self.dense2 = nn.Linear(40, 20)
        self.dense3 = nn.Linear(20, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.dense3(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Finance_001_Model().to(device)

Filename = "./Spark_0630.pt"

if device == 'cpu':
    model.load_state_dict(torch.load(Filename, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(Filename))

def predict(data):
    # Ensure the data has 60 days
    data = data[-60:]
    
    # Normalize and convert data to tensor
    datat = []
    for i in range(60):
        datat.append(torch.tensor([100000*data[i][0], 100000*data[i][1], 100000*data[i][2], 100000*data[i][3]], dtype=torch.float32))
    datat = torch.stack(datat)
    datat = torch.unsqueeze(datat, 0)
    
    # Predict using the model
    model.eval()
    with torch.inference_mode():
        output = model(datat.to(device))
        output = output.tolist()[0]
        output = tuple(val / 100000 for val in output)  # Normalize the output
        
    return output

# get the data from yfinance
import yfinance as yf

ticker = yf.Ticker('KO')
data = ticker.history(interval='1d', period='max', auto_adjust=True)
Open = list(data['Open'])
Close = list(data['Close'])
High = list(data['High'])
Low = list(data['Low'])

data = []
for i in range(len(Open)):
    data.append((Open[i], High[i], Low[i], Close[i]))

real_from_120_to_60 = data[-120:-58]
real_close_from_120_to_60 = [val[3] for val in real_from_120_to_60]
real_from_60_to_0 = data[-60:]
predicted_from_60_to_0 = []
fake_data = real_from_120_to_60.copy()
for i in range(60):
    fake_data.append(predict(fake_data))
    predicted_from_60_to_0.append(fake_data[-1][3])

from matplotlib import pyplot as plt
plt.plot([i for i in range(62)], real_close_from_120_to_60, label='Real')
plt.plot([i for i in range(61, 121)], predicted_from_60_to_0, label='Predicted')
plt.plot([i for i in range(61, 121)], [val[3] for val in real_from_60_to_0], label='Real')
plt.show()