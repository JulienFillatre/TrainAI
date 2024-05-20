# TrainAI

Ce projet utilise le modèle YOLO (You Only Look Once) pour le comptage en temps réel des personnes entrants et sortants d'un train à partir d'un flux vidéo. L'application est construite avec Streamlit pour fournir une interface web interactive.


## Fonctionnalités

- **Comptage en Temps Réel** : Comptage des personnes entrants et sortants d'un train en temps réel en utilisant les modèles YOLO.
- **Sources Vidéo Multiples** : Supporte l'entrée vidéo depuis des flux YouTube ou des liens vidéo locaux.
- **Paramètres Personnalisables** : Ajustez la confiance du modèle, l'IOU, le contraste, la luminosité et les classes de détection.
- **Options de Visualisation** : Visualisez la vidéo traitée avec les détections et la vidéo originale.

## Technologies Utilisées

- **Streamlit** : Pour créer une interface web interactive et conviviale.
- **YOLO** : Pour la détection d'objets en temps réel.
- **OpenCV** : Pour le traitement d'images et de vidéos.
- **VidGear** : Pour le streaming vidéo depuis différentes sources.
- **Python** : Langage de programmation principal utilisé pour le développement du projet.

## Installation

1. Clonez ce dépôt :
    ```bash
    git clone https://github.com/JulienFillatre/TrainAI.git
    cd TrainAI
    ```

2. Créez un environnement virtuel :
    ```bash
    python -m venv env
    ```

3. Activez l'environnement virtuel :
    - Sur Windows :
      ```bash
      .\env\Scripts\activate
      ```
    - Sur macOS et Linux :
      ```bash
      source env/bin/activate
      ```

4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |



1. Lancez l'application Streamlit :
    ```bash
    streamlit run app.py
    ```

2. Configurez les paramètres via l'interface web :
    - **Type de Lien** : Choisissez entre un flux YouTube ou un lien local.
    - **Lien vers le flux vidéo** : Entrez l'URL du flux vidéo.
    - **Choix du modèle YOLO** : Sélectionnez le modèle YOLO à utiliser.
    - **Processeur de Calcul** : Choisissez le dispositif de calcul (CPU, MPS, CUDA).
    - **Confiance du Modèle** : Ajustez le seuil de confiance du modèle.
    - **IOU** : Ajustez le seuil d'Intersection Over Union.
    - **Contrast et Brightness** : Ajustez le contraste et la luminosité de la vidéo.
    - **Sélection des Classes** : Sélectionnez les classes à détecter.
    - **Résolution Vidéo** : Choisissez la résolution de traitement et d'affichage.
    - **Afficher le stream original** : Choisissez d'afficher en plus la vidéo originale.

3. Lancez la détection ou le comptage via les boutons correspondants.

## Code Principal

Le projet comprend deux fonctions principales pour la détection et le comptage d'objets :

- **Visualize** : Lance la détection des classes sélectionnées et active le tracking des résultats en temps réel.
- **Counter** : Compte les personnes entrants et sortants du train en utilisant une ligne de comptage virtuelle.

Ces fonctions sont intégrées à l'application Streamlit et utilisent OpenCV et VidGear pour le traitement et le streaming vidéo.
