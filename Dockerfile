# Usa uma imagem base Python leve
FROM python:3.8-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto para dentro do container
COPY . /app

# Cria e ativa um ambiente virtual, atualiza o pip, instala as dependências e a versão CPU-only do torch
RUN python -m venv env && \
    . env/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Define as variáveis de ambiente padrão para o cliente.
# Estas podem ser sobrescritas na definição da Task do ECS.
ENV SERVER_URL="http://localhost:5000"

# Comando para iniciar o cliente.
# O script main.py no modo client deve usar as variáveis de ambiente.
CMD [ "sh", "-c", ". env/bin/activate && python main.py --mode client --server $SERVER_URL" ]
