import torch.nn as nn
import time
class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        
        self.avgpool = nn.Sequential(         
                   
            nn.AvgPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(4, 4, 3, 1, 0),     
            nn.ReLU(),                      
            nn.AvgPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(100, 10)
    def forward(self, x):
        avg_start = time.time()
        x = self.avgpool(x)
        avg_end = time.time()
        print("avg pool time:", avg_end-avg_start)
        conv_start = time.time()
        x = self.conv2(x)
        conv_end = time.time()
        print("conv time:", conv_end-conv_start)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1) 
        fc_start = time.time()
        output = self.out(x)
        fc_end = time.time()
        print("fc time:", fc_end-fc_start)
        return output, x    # return x for visualization


