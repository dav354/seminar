<p align="center">
 <a href="https://github.com/dav354/seminar/actions">
    <img src="https://github.com/dav354/seminar/actions/workflows/docker_build.yml/badge.svg?branch=main"
         alt="Build Status" />
  </a>
<br/>
  <img src="assets/logo_white.png"  
    alt="Logo"
    width="400" />
</p>

# Schnick Scnack Schnuck Pepper

## Schitt 0
- thema definieren
- api pepper anschauen
- ai vision modell finden (gesten)
- python pepper bibliothek

## Schritt 1 (prototyp)
- Kamera
- Pi mit Coral TPU
- Gestenerkennung der Hand
- Schnick Schack Schnuck business logik

## Schritt 2
- mit pepper zum laufen bekommen
- compute immer noch auf pi & coral tpu
- pepper kameras nutzen
- pepper und pi via wifi verbinden

## Todo

![image](assets/1.png)

- [x] gesture recognition
- [ ] other/custom model for rock, paper, scissors
- [ ] game logic
- [ ] video stream
- [ ] tflight models

## Venv

```shell
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Prerequisites
- install python 3.10

https://coral.ai/docs/edgetpu/compiler/

https://coral.ai/docs/accelerator/get-started/#install-the-edge-tpu-runtime

## Install edge tpu repo

```shell
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install edgetpu-compiler
```

