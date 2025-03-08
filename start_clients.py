import subprocess
import time
import urllib.parse


def start_clients_for_server(server_address, base_id, num_clients):
    processes = []
    for i in range(num_clients):
        client_id = base_id + i
        cmd = [
            "python", "main.py",
            "--mode", "client",
            "--id", str(client_id),
            "--server", server_address
        ]
        print(f"[INFO] Iniciando cliente {client_id} para {server_address}")
        # Inicia o cliente em background
        p = subprocess.Popen(cmd)
        processes.append(p)
        # Pequeno delay para evitar sobrecarga instantânea
        time.sleep(0.2)
    return processes

def get_port_from_url(url):
    url = url.strip()  # Remove espaços extras, se houver
    parsed = urllib.parse.urlparse(url)
    return parsed.port

if __name__ == "__main__":
    # Lista de servidores (endereços completos com IP e porta)
    server_urls = [
        'http://54.232.43.48:5000',
        'http://56.124.104.22:5001',
        # 'http://56.124.27.146:5002'
        # Adicione outros se necessário, como 5003 e 5004
    ]

    all_processes = []
    
    for url in server_urls:
        server_address = url.strip()
        port = get_port_from_url(server_address)
        # Define uma base para o ID do cliente para cada servidor (ex: 100 para porta 5000, 200 para 5001, etc.)
        base_id = (port - 5000 + 1) * 100  
        procs = start_clients_for_server(server_address, base_id, 10)
        all_processes.extend(procs)
    
    # Opcional: aguarda todos os processos finalizarem
    for proc in all_processes:
        proc.wait()
