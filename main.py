# pylint: disable-all
import argparse
import numpy as np
from client import Client
from server import Server
from task import get_model_parameters

def start_client(client_id, server_address):  
    print(f"[INFO] Iniciando cliente {client_id}")
    client = Client(client_id, server_address)
    client.start()

def start_server(server_id, port, peer_servers, min_clients):
    print(f"[INFO] Servidor {server_id} requer {min_clients} clientes por rodada")
    server = Server(server_id, port, peer_servers, min_clients)
    server.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client"], help="Modo de execução: 'server' ou 'client'")
    parser.add_argument("--id", type=int, required=True, help="ID do servidor/cliente")
    parser.add_argument("--port", type=int, help="Porta do servidor")
    parser.add_argument("--peers", type=str, nargs="+", help="Lista de peers (servidores) para comunicação")
    parser.add_argument("--server", type=str, help="Endereço do servidor para o cliente")
    parser.add_argument("--min_clients", type=int, default=2, help="Número mínimo de clientes por rodada")
    parser.add_argument("--rounds", type=int, default=1, help="Número de rodadas de treinamento")

    args = parser.parse_args()

    if args.mode == "server":
        if not args.port or not args.peers:
            raise ValueError("Para iniciar o servidor, forneça --port e --peers.")
        start_server(args.id, args.port, args.peers, args.min_clients)
    elif args.mode == "client":
        if not args.server:
            raise ValueError("Forneça --server")
        start_client(args.id, args.server) 
