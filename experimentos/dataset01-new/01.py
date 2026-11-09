import os
import yaml

# Definir caminhos base
base_dir = os.path.abspath("dataset")
images_dir = os.path.join(base_dir, "test", "ANTIGA")

# Ajuste conforme o nome exato das suas classes e IDs no dataset
classes = ["placa_antiga"] 

# Conteúdo para o arquivo data.yaml
yaml_data = {
    'path': base_dir,
    'train': 'test/ANTIGA',  # Se usar todas para treino rápido no POC
    'val': 'test/ANTIGA',
    'names': {i: name for i, name in enumerate(classes)}
}

yaml_path = os.path.join(base_dir, "data.yaml")

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False)

print(f"Configuração salva em: {yaml_path}")