
import torch.nn as nn
import time

class LCNN(nn.Module):
    def __init__(self):
        super(LCNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=4,            
                kernel_size=3,              
                stride=1,                   
                padding=0,                  
            ),                              
            nn.ReLU(),                        
        )
        
    def forward(self, x):
        start = time.time()
        output = self.conv1(x)
        end = time.time()
        print("First conv:", end-start)
        return output, x    # return x for visualization



        