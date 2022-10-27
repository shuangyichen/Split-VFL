import argparse
from client0 import VFLClientManager
from server import VFLServerManager



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--party_idx', type=int, help='<REQUIRED>')
    parser.add_argument('--client_num', type=int, required=False,default=2,help='<REQUIRED>')
    parser.add_argument('--grpc_ipconfig_path', type=str,help='config table containing ipv4 address of grpc server')
    args = parser.parse_args()

    index = args.party_idx
    if index==0:
        server_manager = VFLServerManager(args,None,index,args.client_num)
        server_manager.run()
    else:
        client_manager = VFLClientManager(args,None,index,args.client_num)
        client_manager.forward()
    