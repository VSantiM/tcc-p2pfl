import subprocess
import time

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

if __name__ == "__main__":
    # Lista de servidores (endereços com as respectivas portas)
    server_ports = [5000, 5001, 5002, 5003, 5004]
    all_processes = []
    
    for port in server_ports:
        server_address = f"http://localhost:{port}"
        # Define uma base para o ID do cliente para cada servidor (por exemplo, 100 para 5000, 200 para 5001, etc.)
        base_id = (port - 5000 + 1) * 100  
        procs = start_clients_for_server(server_address, base_id, 10)
        all_processes.extend(procs)
    
    # Opcional: aguarda todos os processos finalizarem
    for proc in all_processes:
        proc.wait()
