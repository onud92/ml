## Installation
Zuerst die Schritte im Getting Started Guide folgen:  
https://docs.zephyrproject.org/latest/develop/getting_started/index.html

Anschliessend folgende Befehle ausführen:
```shell
# Erstelle einen Workspace-Ordner
mkdir ml
cd ml

# 2. Initialisiere das Projekt mit West
west init -m git@github.com:onud92/ml.git --mr main

# 3. Lade Zephyr und alle Module herunter
west update
```

## Applikation bauen und flashen
```shell
# Für das Arduino Nano 33 BLE Sense Board
west build -b arduino_nano_33_ble/nrf52840/sense
west flash
```