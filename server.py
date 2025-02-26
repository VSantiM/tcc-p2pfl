# pylint: disable-all
from flask import Flask, request, jsonify
import numpy as np
import requests
import time
import threading
import torch
import random
import logging

from typing import List, Dict, Set
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from task import Net, get_model_parameters, set_model_parameters

app = Flask(__name__)

KDD99_ALL_CLASSES = [
    "normal.", "smurf.", "nmap.", "neptune.", "teardrop.",
    "portsweep.", "satan.", "ipsweep.", "back.", "loadmodule.",
    "warezclient.", "warezmaster.", "ftp_write.", "pod.",
    "guess_passwd.", "multihop.", "land.", "buffer_overflow.",
    "rootkit.", "perl.", "imap.", "phf.", "spy."
]

class_to_idx = {label: i for i, label in enumerate(KDD99_ALL_CLASSES)}

class Server:
    def __init__(self, server_id: int, port: int, initial_peers: List[str], min_clients: int = 2):
        self.server_id = server_id
        self.port = port
        self.direct_peers: Set[str] = set(initial_peers)
        self.secondary_peers: Set[str] = set("")
        self.inactive_peers: Dict[str, float] = {}
        self.min_clients = min_clients
        self.registered_clients = set()
        self.lost_peers: Set[str] = set("")
        self.round_clients = {}  # Armazena os clientes por rodada
        self.current_round = 0
        self.aggregation_buffer = {}
        self.round_active = False
        self.heartbeat_interval = 30
        self.total_rounds = 10
        
        # Configuração do logger
        self.logger = logging.getLogger(f"server_{server_id}")
        handler = logging.FileHandler(f"server_{server_id}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Inicialização do modelo
        self._initialize_model()
        

        # Thread de monitoramento
        self.health_checker = threading.Thread(target=self._peer_health_check)
        self.health_checker.daemon = True
        self.health_checker.start()

    def log_round_info(self, round_number, metrics):
        num_clients = len(self.registered_clients)
        total_rounds = self.total_rounds
        num_peers = len(self.get_full_peer_list())
        
        log_message = (f"Rodada {round_number} concluída | "
                    f"Métricas: {metrics} | "
                    f"Clientes: {num_clients} | "
                    f"Total de rodadas: {total_rounds} | "
                    f"Peers conectados: {num_peers}")
        self.logger.info(log_message)


    def _initialize_model(self):
        data, target = fetch_kddcup99(
            subset=None,
            shuffle=True,
            percent10=True,
            return_X_y=True,
            as_frame=True
        )

        categorical_cols = ["protocol_type", "service", "flag"]
        for col in categorical_cols:
            if data[col].dtype == object:
                data[col] = data[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

        numeric_cols = [c for c in data.columns if c not in categorical_cols]

        self.encoder = OneHotEncoder(
            sparse_output=False,
            dtype=np.float32,
            handle_unknown='ignore'
        )
        self.encoder.fit(data[categorical_cols])

        self.scaler = StandardScaler()
        self.scaler.fit(data[numeric_cols])

        n_numeric = len(numeric_cols)
        n_categorical = sum(len(cat_list) for cat_list in self.encoder.categories_)
        in_features = n_numeric + n_categorical
        num_classes = len(KDD99_ALL_CLASSES)

        self.model = Net(in_features, num_classes)
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_classes = KDD99_ALL_CLASSES

    def _peer_health_check(self):
        while True:            
            time.sleep(self.heartbeat_interval)
            self._update_peer_topology()

    def _update_peer_topology(self):
        active_peers = []
        recconected_peers = []
        for peer in list(self.direct_peers):
            if self._is_peer_active(peer):
                active_peers.append(peer)
                try:
                    response = requests.get(f"{peer}/get_peers", timeout=2)
                    if response.status_code == 200:
                        peer_peers = response.json()["peers"]
                        self.secondary_peers.update(peer_peers)
                except:
                    continue
            else:
                self._handle_failed_peer(peer)
        
        for peer in list(self.lost_peers):
            if self._is_peer_active(peer):
                print(f'[Servidor {self.server_id}] Peer {peer} reiniciado. Atualizando conexão...')
                recconected_peers.append(peer)
                self.lost_peers.discard(peer)
                try:
                    response = requests.get(f"{peer}/get_peers", timeout=2)
                    if response.status_code == 200:
                        peer_peers = response.json()["peers"]
                        self.secondary_peers.update(peer_peers)
                except:
                    continue
            else:
                continue
        
        self.direct_peers = set(active_peers + recconected_peers)
        self._clean_secondary_peers()

    def _handle_failed_peer(self, peer_url: str):
        print(f"[Servidor {self.server_id}] Peer {peer_url} inativo. Tentando reconexão...")
        
        try:
            response = requests.get(f"{peer_url}/get_peers", timeout=2)
            if response.status_code == 200:
                failed_peer_peers = response.json()["peers"]
                new_peers = [p for p in failed_peer_peers 
                           if p != f"http://localhost:{self.port}" and p not in self.direct_peers]
                self.direct_peers.update(new_peers)
        except:
            pass
        
        self.lost_peers.add(peer_url)
        print(f'[DEBUG] Lost peers list {list(self.lost_peers)}')
        self.inactive_peers[peer_url] = time.time()
        self.direct_peers.discard(peer_url)
        self.secondary_peers.discard(peer_url)

    def _clean_secondary_peers(self):
        active_secondary = set()
        for peer in self.secondary_peers:
            if self._is_peer_active(peer):
                active_secondary.add(peer)
        self.secondary_peers = active_secondary

    def _is_peer_active(self, peer_url: str) -> bool:
        try:
            response = requests.get(f"{peer_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_pipeline_info(self):
        encoder_cats_str = []
        for cat_array in self.encoder.categories_:
            cat_list_str = []
            for c in cat_array:
                if isinstance(c, bytes):
                    c = c.decode("utf-8")
                cat_list_str.append(c)
            encoder_cats_str.append(cat_list_str)

        return {
            "encoder_categories": encoder_cats_str,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "target_classes": self.target_classes,
        }

    def evaluate_model(self):
        data, target = fetch_kddcup99(
            subset=None,
            shuffle=True,
            percent10=True,
            return_X_y=True,
            as_frame=True
        )

        # Preparar os dados de teste (utilizando o mesmo pré-processamento do servidor)
        categorical_cols = self.categorical_cols
        numeric_cols = self.numeric_cols

        for col in categorical_cols:
            if data[col].dtype == object:
                data[col] = data[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

        # Aplicar escalonamento e codificação
        data_nums = self.scaler.transform(data[numeric_cols].to_numpy())
        data_cats = self.encoder.transform(data[categorical_cols].to_numpy())
        X = np.hstack([data_nums, data_cats])

        # Mapear os rótulos com base nas classes do servidor
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.target_classes)}
        y_mapped = []
        for row_label in target:
            if isinstance(row_label, bytes):
                row_label = row_label.decode("utf-8")
            if row_label in class_to_idx:
                y_mapped.append(class_to_idx[row_label])

        if not y_mapped:
            print("[Servidor] Nenhum rótulo mapeado para teste.")
            return None

        # Criar DataLoader para avaliação
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Avaliar o modelo utilizando a função calculate_metrics do task.py
        from task import calculate_metrics
        metrics = calculate_metrics(self.model, test_loader)
        print(f"[Servidor {self.server_id}] Resultados da avaliação:")
        print(f'Clientes {list(self.registered_clients)}')
        print(f"  Acurácia: {metrics['accuracy']:.4f}")
        print(f"  Precisão: {metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        return metrics



    def collect_and_aggregate_models(self, received_models):
        if not received_models:
            print("[Servidor] Nenhum modelo para agregar.")
            return

        # Obtém os parâmetros atuais do modelo local
        local_params = get_model_parameters(self.model)
        # Inclui o modelo local para que o FedAvg seja realizado com ele
        models = [local_params] + received_models

        aggregated_parameters = {}
        keys = list(local_params.keys())
        for param_key in keys:
            arrays = [np.array(model[param_key], dtype=np.float32) for model in models]
            mean_arr = np.mean(arrays, axis=0)
            aggregated_parameters[param_key] = mean_arr.tolist()

        set_model_parameters(self.model, aggregated_parameters)
        print(f"[Servidor {self.server_id}] Modelo agregado com sucesso (FedAvg).")


    def start_round(self):
        self.current_round += 1
        self.round_active = True
        self.aggregation_buffer = {}
        print(f"[Servidor {self.server_id}] Rodada {self.current_round} iniciada. Peers ativos: {self.get_full_peer_list()}")
        

    def _communicate_with_peers(self):
        model_params = get_model_parameters(self.model)
        
        for peer in self.direct_peers:
            print('[INFO] Envio de modelo para', peer)
            self._send_to_peer(peer, model_params)

    def _send_to_peer(self, peer_url: str, model_params: dict):
        try:
            requests.post(
                f"{peer_url}/receive_server_model_from_peer",
                json={
                    "model": model_params,
                    "sender_id": self.server_id
                },
                timeout=3
            )
        except:
            print(f"[Servidor {self.server_id}] Falha ao enviar para {peer_url}")

    def get_full_peer_list(self) -> List[str]:
        return list(self.direct_peers.union(self.secondary_peers))

    def start(self):
        create_server_instance(self.server_id, self.port, list(self.direct_peers), self.min_clients)
        print('[INFO] Iniciando servidor Flask...')
        app.run(host="0.0.0.0", port=self.port)

    def stop(self):
        self.running = False

server_instance = None

def create_server_instance(server_id, port, peers, min_clients):
    global server_instance
    server_instance = Server(server_id, port, peers, min_clients)
    print(f"[INFO] Servidor {server_id} inicializado na porta {port} com peers: {peers}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/get_pipeline', methods=['GET'])
def get_pipeline():
    global server_instance
    if server_instance is None:
        return jsonify({"status": "error", "message": "Servidor não inicializado."}), 500
    return jsonify(server_instance.get_pipeline_info()), 200

@app.route('/get_model', methods=['GET'])
def get_model():
    global server_instance
    if server_instance is None:
        return jsonify({"status": "error", "message": "Servidor não inicializado."}), 500
    model_params = get_model_parameters(server_instance.model)
    return jsonify({"model": model_params}), 200

@app.route('/receive_model', methods=['POST'])
def receive_model():
    global server_instance
    if server_instance is None:
        return jsonify({"status": "error", "message": "Servidor não inicializado."}), 500

    data = request.json
    client_id = data.get("client_id")
    model_params = data.get("model")
    current_round = data.get("round")

    if current_round not in server_instance.aggregation_buffer:
        server_instance.aggregation_buffer[current_round] = []
    
    server_instance.aggregation_buffer[current_round].append(model_params)
    server_instance.registered_clients.add(client_id)  # Garante que o cliente está registrado

    # Agrega se o número de modelos recebidos for igual ao número de clientes registrados
    if len(server_instance.aggregation_buffer[current_round]) >= len(server_instance.registered_clients):
        server_instance.collect_and_aggregate_models(server_instance.aggregation_buffer[current_round])
        metrics = server_instance.evaluate_model()
        server_instance.log_round_info(current_round, metrics)
        print(f"[Servidor {server_instance.server_id}] Rodada {current_round} concluída.")
        
        # Somente atualiza round e avalia se houver clientes (servidor primário)
        if server_instance.registered_clients:
            server_instance.current_round += 1
            print(f"[Servidor {server_instance.server_id}] Rodada {server_instance.current_round} iniciada.")
            # Opcional: chamar avaliação
            server_instance.evaluate_model()

        # Se alcançou o total de rounds, pode comunicar aos peers ou resetar a rodada
        if server_instance.current_round > server_instance.total_rounds:
            print(f"[Servidor {server_instance.server_id}] Treinamento finalizado. Enviando modelo para peers.")
            server_instance.current_round = 0
            server_instance.round_active = False
            server_instance._communicate_with_peers()

    return jsonify({"status": "received"}), 200


@app.route('/receive_server_model_from_peer', methods=['POST'])
def receive_server_model_from_peer():
    global server_instance
    if server_instance is None:
        return jsonify({"status": "error", "message": "Servidor não inicializado."}), 500

    data = request.json
    sender_id = data.get("sender_id")
    model_params = data.get("model")

    print(f"[Servidor {server_instance.server_id}] Parâmetros recebidos do servidor {sender_id} - Agregando ao modelo local...")
    # Aqui agregamos apenas o modelo do peer com o modelo local (FedAvg)
    server_instance.collect_and_aggregate_models([model_params])
    print(f"[Servidor {server_instance.server_id}] Modelo agregado do servidor {sender_id}.")

    return jsonify({"status": "received"}), 200

@app.route('/get_peers', methods=['GET'])
def get_peers():
    global server_instance
    return jsonify({
        "peers": list(server_instance.direct_peers),
        "active_peers": server_instance.get_full_peer_list()
    }), 200

@app.route('/receive_server_model', methods=['POST'])
def receive_server_model():
    global server_instance
    data = request.json
    model_params = data.get("model")
    sender_id = data.get("sender_id")
    
    local_params = get_model_parameters(server_instance.model)
    aggregated_params = {
        k: (np.array(local_params[k]) + np.array(model_params[k])) / 2
        for k in local_params.keys()
    }
    set_model_parameters(server_instance.model, aggregated_params)
    
    print(f"[Servidor {server_instance.server_id}] Modelo agregado do servidor {sender_id}")
    return jsonify({"status": "received"}), 200

@app.route('/register', methods=['POST'])
def register_client():
    global server_instance
    data = request.json
    client_id = data.get("client_id")
    
    server_instance.registered_clients.add(client_id)

    # Seleciona backup aleatório dentre os peers diretos (exceto o próprio servidor)
    available_backups = list(server_instance.direct_peers - {f"http://localhost:{server_instance.port}"})
    backup = random.choice(available_backups) if available_backups else ""

    print(f"[Servidor {server_instance.server_id}] Cliente {client_id} registrado com backup {backup}.")
    
    if len(server_instance.registered_clients) >= server_instance.min_clients and not server_instance.round_active:
        server_instance.start_round()
    
    return jsonify({"status": "registered", "backup": backup}), 200

@app.route('/get_rounds', methods=['GET'])
def get_rounds():
    return jsonify({"total_rounds": server_instance.total_rounds})


@app.route('/round_status', methods=['GET'])
def round_status():
    global server_instance
    return jsonify({
        "round_active": server_instance.round_active,
        "current_round": server_instance.current_round
    }), 200

@app.route('/get_backup', methods=['GET'])
def get_backup():
    global server_instance
    # Exclui o próprio endereço do servidor da lista de backups
    available_backups = list(server_instance.direct_peers - {f"http://localhost:{server_instance.port}"})
    if available_backups:
        new_backup = random.choice(available_backups)
        return jsonify({"backup": new_backup}), 200
    else:
        return jsonify({"backup": ""}), 200


@app.route('/evaluate_model', methods=['GET'])
def evaluate_model_route():
    global server_instance
    if server_instance is None:
        return jsonify({"status": "error", "message": "Servidor não inicializado."}), 500
    metrics = server_instance.evaluate_model()
    if metrics is None:
        return jsonify({"status": "error", "message": "Erro na avaliação."}), 500
    return jsonify(metrics), 200


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inicia um servidor federado.")
    parser.add_argument("--id", type=int, required=True, help="ID do servidor")
    parser.add_argument("--port", type=int, required=True, help="Porta do servidor")
    parser.add_argument("--peers", type=str, nargs='+', help="Lista inicial de peers")
    parser.add_argument("--min_clients", type=int, default=2, help="Mínimo de clientes por rodada")

    args = parser.parse_args()
    create_server_instance(args.id, args.port, args.peers, args.min_clients)
    server_instance.start()