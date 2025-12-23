## Installation von Docker

```shell
# 1. Paketliste aktualisieren
sudo apt update

# 2. Docker und Hilfsprogramme installieren
sudo apt install -y docker.io curl

# 3. Docker-Dienst starten und aktivieren (damit er beim Booten angeht)
sudo systemctl start docker
sudo systemctl enable docker

# Ihren Benutzer zur Gruppe 'docker' hinzufügen und PC neu starten
sudo usermod -aG docker $USER

# repo hinzufügen
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# installieren und konfigurieren
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Sagt Docker, dass es NVIDIA nutzen soll
sudo nvidia-ctk runtime configure --runtime=docker

# Startet Docker neu, um die Änderungen zu übernehmen
sudo systemctl restart docker

# Image bauen
docker build -t my_image .

# run 
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace my_image python hello_tf.py
```