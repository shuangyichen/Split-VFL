import torch

from torchvision import datasets
from torchvision.transforms import ToTensor
from model.server_model import SCNN
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import argparse
from distributed.communication.message import Message
from distributed.server.server_manager import ServerManager
from message_define import MyMessage
import numpy as np
import time
class VFLServerManager(ServerManager):
    def __init__(self,args,comm=None,rank=0, size=0,backend="GRPC"):
        super().__init__(args, comm, rank, size, backend)

    #Load the data
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )
        self.train = train_data
        self.test = test_data
        
        self.idx = args.party_idx
        self.finished_inference = 0
        self.num_clients = args.client_num
        self.collector_inter_res = [None]*self.num_clients

        #Load the model
        state_dict = torch.load('mh_cnn_1')
        server_nn = SCNN()
        with torch.no_grad():
            server_nn.conv2[0].weight.copy_(state_dict['conv2.0.weight'])
            server_nn.conv2[0].bias.copy_(state_dict['conv2.0.bias'])
            server_nn.out.weight.copy_(state_dict['out.weight'])
            server_nn.out.bias.copy_(state_dict['out.bias'])
        self.model = server_nn


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_INTERMEDIATE_RESULT,self.handle_message_intermediate_result)

        #Forward
    def handle_message_intermediate_result(self,msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        print("Receiving results from client ",sender_id)
        mid_res =  msg_params.get(MyMessage.MSG_ARG_INTER_RES)
        self.collector_inter_res[sender_id-1] = mid_res
        received = 0
        for i in range(self.num_clients):
            if self.collector_inter_res[i]!=None:
                received += 1
        if received == self.num_clients:
            self.server_side_forward()
            

    def server_side_forward(self):
        self.model.eval()
        images, labels = self.test[self.finished_inference]
            #Receiving intermediate result from clients
        for i in range(self.num_clients):
            tmp = torch.from_numpy(np.array(self.collector_inter_res[i],).reshape(-1, 1))
            self.collector_inter_res[i] = torch.reshape(tmp,(4,12,26))
            # print(self.collector_inter_res[i].size())
        input = torch.cat((self.collector_inter_res[0], self.collector_inter_res[1]), 1)
        input = input[None,:,:,:]
        # print(input.size())
        test_output,last_layer = self.model(input.float())
        prediction=np.argmax(test_output.detach().numpy())
        print("Label is ",labels)
        print("Prediction is ", prediction)

        # pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # accuracy = (pred_y == labels).sum().item() / float(labels.size(0))

