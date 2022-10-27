import torch

from torchvision import datasets
from torchvision.transforms import ToTensor
from model.local_model import LCNN
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from distributed.communication.message import Message
from distributed.client.client_manager import ClientManager
import argparse
from message_define import MyMessage
import time
class VFLClientManager(ClientManager):
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
        # self.batch_size = args.batch_size
        self.train = train_data
        self.test = test_data
       
        self.idx = args.party_idx
        self.finished_inference = 0
        
        model_name = "localnn" + str(self.idx)
        #Load the model
        state_dict = torch.load('mh_cnn_1')
        local = LCNN()
        with torch.no_grad():
            local.conv1[0].weight.copy_(state_dict[model_name+'.0.conv1.0.weight'])
            local.conv1[0].bias.copy_(state_dict[model_name+'.0.conv1.0.bias'])
        self.model = local
        
    def register_message_receive_handlers(self):
        pass

    #MSG_TYPE_C2S_PHASE1_DONE
        #Forward
    def forward(self):
        self.model.eval()
        # with torch.no_grad():
        #     for images, labels in loaders['test']:
        images, labels = self.test[self.finished_inference]
        images = images[:,14*(self.idx-1):14*self.idx,:]
        images = images[None,:,:,:]
        output, last_layer = self.model(images)
        self.finished_inference += 1
        output_list = output.detach().numpy().tolist()
        self.send_intermediate_result_to_server(output_list)

    def send_intermediate_result_to_server(self, intermedaite_result):
        message = Message(MyMessage.MSG_TYPE_C2S_INTERMEDIATE_RESULT, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_INTER_RES, intermedaite_result)
        self.send_message(message)
        # print("Finished")
