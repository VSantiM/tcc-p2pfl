# pylint: disable-all
import torch
import requests
import numpy as np
import time

from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from task import Net, get_model_parameters, set_model_parameters, train_model, calculate_metrics


class Client:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.backup_url = ''
        self.rounds = 1 # Número de rodadas (inicializado com 1)
        self.current_server_round = 1 # Rodada atual do servidor
        self.round_atual = 0
        self.register_with_server()

        # 1) Obter pipeline do servidor: OneHot, Scaler e 'target_classes'
        pipeline_info = self.request_pipeline()
        if pipeline_info is None:
            raise RuntimeError(f"[Cliente] Falha ao obter pipeline (servidor/backup).")

        encoder_categories = pipeline_info["encoder_categories"]
        scaler_mean = pipeline_info["scaler_mean"]
        scaler_scale = pipeline_info["scaler_scale"]
        global_classes = pipeline_info["target_classes"]

        # 2) Carregar dados brutos do KDD99 localmente
        data, target = fetch_kddcup99(
            subset=None,
            shuffle=True,
            percent10=True,
            return_X_y=True,
            as_frame=True
        )

        # 2.1) Decodificar colunas categóricas 
        categorical_cols = ["protocol_type", "service", "flag"]
        for col in categorical_cols:
            if data[col].dtype == object:
                data[col] = data[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

        numeric_cols = [c for c in data.columns if c not in categorical_cols]

        # 3) Reconstruir o OneHotEncoder com categorias do servidor (CORREÇÃO APPLICADA)
        self.encoder = OneHotEncoder(
            sparse_output=False,
            dtype=np.float32,
            handle_unknown='ignore',
            categories=encoder_categories
        )
        
        # Ajuste com dados dummy para definir o número de features
        dummy_data = [[""] * len(encoder_categories)]  # Número de colunas = 3
        self.encoder.fit(dummy_data)

        # 4) Reconstruir o StandardScaler com parâmetros do servidor
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(scaler_mean, dtype=np.float32)
        self.scaler.scale_ = np.array(scaler_scale, dtype=np.float32)
        self.scaler.n_features_in_ = len(self.scaler.mean_)

        # 5) Aplicar transformações (CORREÇÃO PARA WARNINGS)
        data_nums = self.scaler.transform(data[numeric_cols].to_numpy())
        data_cats = self.encoder.transform(data[categorical_cols].to_numpy())
        X = np.hstack([data_nums, data_cats])

        # 6) Mapear rótulos locais para índices globais
        class_to_idx = {cls_name: i for i, cls_name in enumerate(global_classes)}
        y_mapped = []
        for row_label in target:
            if isinstance(row_label, bytes):
                row_label = row_label.decode("utf-8")
            if row_label in class_to_idx:
                y_mapped.append(class_to_idx[row_label])

        # 7) Montar DataLoader local
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        self.local_data = DataLoader(dataset, batch_size=32, shuffle=True)

        # 8) Criar modelo local
        in_features = X.shape[1]
        num_classes = len(global_classes)
        self.model = Net(in_features, num_classes)

        # 9) Obter modelo inicial do servidor
        self.request_model()

    # ---------- MÉTODOS DE REGISTRO ----------
    def register_with_server(self):
        """Registra o cliente no servidor principal."""
        try:
            resp = requests.post(
                f"{self.server_url}/register",
                json={"register": True},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                backup = data.get("backup")
                if backup:
                    self.backup_url = backup
                print(f"[Cliente] Registrado com backup: {self.backup_url}")
            else:
                self.register_with_backup()
        except requests.exceptions.RequestException:
            self.register_with_backup()

    def register_with_backup(self):
        """Tenta registro no servidor de backup."""
        try:
            resp = requests.post(
                f"{self.backup_url}/register",
                json={"register": True},
                timeout=15
            )
        except requests.exceptions.RequestException:
            print(f"[Cliente] Falha ao registrar-se no servidor de backup.")

    def update_backup_from_server(self):
        """Solicita ao servidor principal um novo backup."""
        try:
            print(f"[Cliente] Solicitando novo backup ao servidor principal")
            resp = requests.get(f"{self.server_url}/get_backup", timeout=15)
            if resp.status_code == 200:
                new_backup = resp.json().get("backup")
                if new_backup:
                    print(f"[Cliente] Novo backup obtido do servidor principal: {new_backup}")
                    self.backup_url = new_backup
                    return True
        except requests.exceptions.RequestException as e:
            print(f"[Cliente] Falha ao solicitar backup do servidor principal: {e}")
        return False

    def check_backup_health(self):
        """Realiza health check no backup; se falhar, tenta atualizar o backup."""
        try:
            resp = requests.get(f"{self.backup_url}/health", timeout=3)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            print(f"[Cliente] Backup {self.backup_url} falhou no health check.")
            self.update_backup_from_server()
        return False


    # ---------- MÉTODOS DE COMUNICAÇÃO ----------
    def request_pipeline(self):
        """Tenta obter pipeline do servidor, se falhar, do backup."""
        try:
            resp = requests.get(f"{self.server_url}/get_pipeline", timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass

        try:
            resp = requests.get(f"{self.backup_url}/get_pipeline", timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass

        return None

    def request_model(self):
        """Obtém o modelo inicial do servidor."""
        try:
            resp = requests.get(f"{self.server_url}/get_model", timeout=15)
            if resp.status_code == 200:
                model_params = resp.json().get("model")
                self.receive_model(model_params)
                print(f"[Cliente] Modelo inicial recebido (principal).")
            else:
                raise ValueError(f"Erro ao requisitar modelo. Status: {resp.status_code}")
        except requests.exceptions.RequestException:
            print(f"[Cliente] Falha ao requisitar modelo do principal. Tentando backup.")
            self.check_backup_health()  # Verifica se o backup está saudável
            self.request_model_from_backup()

    def request_backup_model(self):
        """Tenta requisitar o modelo a partir do backup atualizado."""
        try:
            resp = requests.get(f"{self.backup_url}/get_model", timeout=15)
            if resp.status_code == 200:
                model_params = resp.json().get("model")
                self.receive_model(model_params)
                print(f"[Cliente] Modelo inicial recebido (backup).")
            else:
                raise ValueError(f"Erro ao requisitar modelo do backup. Status: {resp.status_code}")
        except requests.exceptions.RequestException:
            print(f"[Cliente] Falha ao requisitar modelo do backup.")

    # ---------- MÉTODOS DE MODELO ----------
    def receive_model(self, model_params):
        """Recebe parâmetros do modelo e carrega localmente."""
        if isinstance(model_params, dict):
            model_params = {k: torch.tensor(v).float() for k, v in model_params.items()}
        elif isinstance(model_params, (list, np.ndarray)):
            model_params = {
                name: torch.tensor(param).float()
                for name, param in zip(self.model.state_dict().keys(), model_params)
            }
        else:
            raise ValueError(f"Formato inválido: {type(model_params)}")

        set_model_parameters(self.model, model_params)
        print(f"[Cliente] Modelo atualizado com sucesso.")

    def train_model_local(self):
        """Treina localmente e retorna parâmetros + métricas."""
        train_model(self.model, self.local_data, epochs=10, learning_rate=0.001)
        metrics = calculate_metrics(self.model, self.local_data)
        print(f"[Cliente] Treino concluído. Acurácia: {metrics['accuracy']}")
        return get_model_parameters(self.model), metrics

    # ---------- LÓGICA DE TREINAMENTO ----------
    def send_model(self):
        """Envia modelo + métricas para o servidor."""
        trained_params, metrics = self.train_model_local()
        data = {
            "model": trained_params,
            "metrics": metrics,
            "round": self.round_atual
        }
        
        try:
            resp = requests.post(f"{self.server_url}/receive_model", json=data, timeout=300)
            if resp.status_code == 200:
                print("[Cliente] Modelo enviado ao principal.")
            else:
                print(f"[Cliente] Erro ao enviar (status={resp.status_code}), tentando backup.")
                self.handle_failover(trained_params)
        except requests.exceptions.RequestException:
            print("[Cliente] Erro de conexão, tentando backup.")
            self.handle_failover(trained_params)

    def handle_failover(self, trained_params):
        """Tenta enviar para o servidor de backup."""
        data = {"model": trained_params}
        try:
            resp = requests.post(f"{self.backup_url}/receive_model", json=data, timeout=15)
            if resp.status_code == 200:
                print(f"[Cliente] Modelo enviado ao backup.")
        except requests.exceptions.RequestException:
            print(f"[Cliente] Falha ao enviar para backup.")

    def get_server_rounds(self):
        try:
            resp = requests.get(f"{self.server_url}/get_rounds", timeout=15)
            if resp.status_code == 200:
                return resp.json().get("current_round", 1), resp.json().get("total_rounds", 1)
        except requests.exceptions.RequestException:
            print(f"[Cliente] Falha ao obter rodadas do servidor.")
        return 1  # Caso falhe, assume 1 rodada por segurança

    def start(self):
        """Executa o ciclo completo de treinamento."""

        round_atual_servidor, self.rounds = self.get_server_rounds()

        for round in range(round_atual_servidor, self.rounds + 1):
            self.round_atual = round
            print(f"\n[Cliente] Rodada {round}/{self.rounds}")
            
            # Aguardar início da rodada
            self._wait_for_round_start()
            
            # Ciclo completo
            self.request_model()
            self.send_model()

    def _wait_for_round_start(self):
        """Aguarda até que a rodada atual esteja ativa."""
        while True:
            try:
                resp = requests.get(f"{self.server_url}/round_status", timeout=5)
                if resp.json().get("round_active") and resp.json().get("current_round") == self.round_atual:
                    print(f'[DEBUG] Cliente iniciando rodada {self.round_atual}')
                    return
                time.sleep(5)
            except requests.exceptions.RequestException:
                pass  # Tentar novamente


def start_client(server_address: str, backup_address: str):  
    print(f"[INFO] Iniciando cliente")
    client = Client(server_address, backup_address)
    client.start()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inicia um cliente federado.")
    parser.add_argument("--server", type=str, required=True, help="Endereço do servidor")
    parser.add_argument("--backup", type=str, required=True, help="Endereço de backup")

    args = parser.parse_args()
    start_client(args.id, args.server, args.backup)