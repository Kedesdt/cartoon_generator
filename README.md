# Gerador de Cartoon com IA

Aplicação Flask que transforma fotos em cartoon usando ControlNet + Stable Diffusion.

## 🎨 Características

- Upload de imagens (PNG, JPG, JPEG, WEBP)
- 4 estilos diferentes:
  - Cartoon (Desenho Animado)
  - Anime/Mangá
  - Quadrinhos (Comic Book)
  - Aquarela
- Interface web responsiva e amigável
- Processamento local (open source)

## 🚀 Instalação

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

**Nota:** A instalação pode demorar alguns minutos devido ao tamanho dos pacotes.

### 2. Requisitos de Sistema

**⚠️ SDXL requer mais recursos que SD 1.5:**

- **GPU (OBRIGATÓRIA):** NVIDIA com 10GB+ VRAM (RTX 3080 ou superior)
- **CPU:** Não recomendado (muito lento - 5-10 minutos por imagem)
- **RAM:** Mínimo 16GB (32GB recomendado)
- **Espaço em disco:** ~13GB para modelos SDXL

## 💻 Como Usar

### 1. Executar a aplicação

```bash
python app.py
```

### 2. Acessar no navegador

Abra: `http://localhost:5000`

### 3. Processo

1. Clique em "Escolher Foto" e selecione uma imagem
2. Escolha o estilo desejado
3. Clique em "Gerar Cartoon"
4. Aguarde 30-60 segundos (primeira vez pode demorar mais)
5. Veja o resultado e faça download se desejar

## 📋 Primeira Execução

Na primeira vez que você executar, os modelos serão baixados automaticamente:

- **ControlNet SDXL Canny** (~5GB)
- **Stable Diffusion XL 1.0** (~7GB)
- **Total:** ~12-13GB

⏱️ **Tempo de download:** 15-30 minutos (depende da internet)

Isso é feito uma única vez. Nas próximas execuções, os modelos já estarão salvos.

## 🛠️ Tecnologias

- **Flask:** Framework web
- **ControlNet SDXL:** Controle de geração de imagem (versão XL)
- **Stable Diffusion XL:** Modelo de geração de alta qualidade
- **Diffusers (Hugging Face):** Pipeline de processamento
- **PyTorch:** Framework de deep learning

## 📝 Estrutura do Projeto

```
image_generator/
├── app.py                 # Aplicação Flask
├── model.py              # Lógica do ControlNet
├── requirements.txt      # Dependências
├── templates/
│   └── index.html       # Interface web
├── static/
│   ├── css/
│   │   └── style.css    # Estilos
│   ├── uploads/         # Imagens enviadas
│   └── outputs/         # Cartoons gerados
└── README.md
```

## ⚠️ Resolução de Problemas

### GPU não detectada

- Verifique se CUDA está instalado: `nvidia-smi`
- Instale PyTorch com CUDA: https://pytorch.org/get-started/locally/

### Memória insuficiente

- Feche outros programas
- Use imagens menores (máx 768px)
- A aplicação já otimiza automaticamente

### Erro ao baixar modelos

- Verifique sua conexão com a internet
- Pode precisar de VPN se houver restrição regional
- Os modelos são baixados de huggingface.co

## 📄 Licença

Open Source - Livre para uso pessoal e comercial
