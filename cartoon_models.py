import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import CannyDetector
from PIL import Image
import numpy as np
import cv2
import os
from torchvision import transforms


def setup_local_cache():
    """Configura cache local do Hugging Face na pasta do projeto"""
    # Obter diretório do projeto
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(project_dir, "huggingface")
    
    # Criar diretório se não existir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Configurar variáveis de ambiente para HF cache
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    
    print(f"Cache configurado para: {cache_dir}")
    return cache_dir


class CartoonGenerator:
    def __init__(self):
        # Configurar cache local antes de qualquer carregamento
        setup_local_cache()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.canny_detector = None
        print(f"Usando dispositivo: {self.device}")

    def load_model(self):
        """Carrega os modelos ControlNet e Stable Diffusion"""
        if self.pipe is not None:
            return

        print("Carregando modelos... Isso pode demorar alguns minutos na primeira vez.")

        # Carregar ControlNet Canny
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Carregar Stable Diffusion com ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )

        # Otimizações
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)

        # Habilitar otimizações de memória
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            # self.pipe.enable_xformers_memory_efficient_attention()

        # Detector Canny
        self.canny_detector = CannyDetector()

        print("Modelos carregados com sucesso!")

    def process_image(self, image_path, output_path, style="cartoon"):
        """
        Processa a imagem e transforma em cartoon

        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho para salvar a imagem gerada
            style: Estilo do cartoon (cartoon, anime, comic, etc)
        """
        self.load_model()

        # Carregar e preparar imagem
        input_image = Image.open(image_path).convert("RGB")

        # Redimensionar para melhor performance (máximo 768px)
        max_size = 768
        ratio = min(max_size / input_image.width, max_size / input_image.height)
        if ratio < 1:
            new_size = (int(input_image.width * ratio), int(input_image.height * ratio))
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)

        # Detectar bordas com Canny
        print("Detectando bordas...")
        canny_image = self.canny_detector(input_image)

        # Prompts baseados no estilo escolhido
        style_prompts = {
            "cartoon": "cartoon style, vibrant colors, bold outlines, cel shaded, animation style, colorful, detailed, high quality",
            "anime": "anime style, manga art, vibrant colors, detailed eyes, japanese animation, high quality",
            "comic": "comic book style, pop art, bold colors, halftone dots, graphic novel, detailed",
            "watercolor": "watercolor painting, soft colors, artistic, painterly, beautiful, detailed",
        }

        prompt = style_prompts.get(style, style_prompts["cartoon"])
        negative_prompt = "ugly, blurry, low quality, distorted, deformed, bad anatomy, watermark, text"

        print(f"Gerando imagem no estilo {style}...")

        # Gerar imagem
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.8,
            ).images[0]

        # Salvar resultado
        result.save(output_path)
        print(f"Imagem salva em: {output_path}")

        return output_path

    def cleanup(self):
        """Libera memória"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Instância global
generator = CartoonGenerator()


class CartoonGANGenerator:
    """Gerador rápido usando AnimeGAN2 (CartoonGAN)"""
    
    def __init__(self):
        # Configurar cache local antes de qualquer carregamento
        setup_local_cache()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        print(f"CartoonGAN usando dispositivo: {self.device}")
    
    def preprocess_image(self, img):
        """
        Melhora a qualidade da imagem de entrada antes do processamento
        """
        from PIL import ImageEnhance, ImageFilter
        
        # Aplicar um filtro suave para reduzir ruído
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
        
        # Ajustar brilho automaticamente se a imagem estiver muito escura ou clara
        import numpy as np
        img_array = np.array(img)
        brightness = np.mean(img_array)
        
        if brightness < 100:  # Imagem muito escura
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)
        elif brightness > 200:  # Imagem muito clara
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.9)
        
        return img
    
    def load_model(self, style="face_paint_512_v2"):
        """
        Carrega modelo AnimeGAN2
        
        Estilos disponíveis:
        - face_paint_512_v2: Cartoon ocidental moderno
        - paprika: Anime vibrante e colorido
        - hayao: Estilo Studio Ghibli
        - shinkai: Estilo Makoto Shinkai (Your Name)
        """
        if style in self.models:
            return self.models[style]
        
        print(f"Carregando modelo CartoonGAN ({style})...")
        
        try:
            model = torch.hub.load(
                'bryandlee/animegan2-pytorch:main',
                'generator',
                pretrained=style,
                device=self.device,
                progress=True
            )
            model.eval()
            self.models[style] = model
            print(f"Modelo {style} carregado!")
            return model
        except Exception as e:
            print(f"Erro ao carregar {style}: {e}")
            # Fallback para modelo padrão
            if style != "face_paint_512_v2":
                print("Usando modelo padrão face_paint_512_v2")
                return self.load_model("face_paint_512_v2")
            raise
    
    def process_image(self, image_path, output_path, style="face_paint_512_v2"):
        """
        Processa imagem rapidamente com CartoonGAN
        
        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho para salvar
            style: Estilo do cartoon (face_paint_512_v2, paprika, hayao, shinkai)
        """
        model = self.load_model(style)
        
        # Carregar imagem
        img = Image.open(image_path).convert('RGB')
        
        # Pré-processar imagem para melhor qualidade
        img = self.preprocess_image(img)
        
        original_size = img.size
        
        # Calcular novo tamanho mantendo aspect ratio
        # Redimensionar para que a maior dimensão seja 512
        max_size = 512
        ratio = min(max_size / img.width, max_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        # Redimensionar mantendo proporção
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Criar imagem quadrada com padding
        square_img = Image.new('RGB', (max_size, max_size), (255, 255, 255))
        offset_x = (max_size - new_width) // 2
        offset_y = (max_size - new_height) // 2
        square_img.paste(img_resized, (offset_x, offset_y))
        
        # Preparar transformações com normalização adequada
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Converter para tensor
        img_tensor = transform(square_img).unsqueeze(0)
        
        # Mover para device
        img_tensor = img_tensor.to(self.device)
        
        print(f"Gerando cartoon no estilo {style}...")
        
        # Gerar cartoon
        with torch.no_grad():
            cartoon_tensor = model(img_tensor)
        
        # Desnormalizar e converter de volta para imagem
        cartoon_tensor = cartoon_tensor.cpu().squeeze(0)
        # Desnormalizar
        cartoon_tensor = cartoon_tensor * 0.5 + 0.5
        # Clampar valores entre 0 e 1
        cartoon_tensor = torch.clamp(cartoon_tensor, 0, 1)
        
        cartoon_img = transforms.ToPILImage()(cartoon_tensor)
        
        # Extrair a região sem padding
        cartoon_cropped = cartoon_img.crop((
            offset_x, 
            offset_y, 
            offset_x + new_width, 
            offset_y + new_height
        ))
        
        # Redimensionar de volta ao tamanho original
        cartoon_final = cartoon_cropped.resize(original_size, Image.Resampling.LANCZOS)
        
        # Aplicar melhoria de qualidade: ajustar contraste e saturação
        from PIL import ImageEnhance
        
        # Melhorar contraste ligeiramente
        enhancer = ImageEnhance.Contrast(cartoon_final)
        cartoon_final = enhancer.enhance(1.1)
        
        # Melhorar saturação ligeiramente
        enhancer = ImageEnhance.Color(cartoon_final)
        cartoon_final = enhancer.enhance(1.15)
        
        # Salvar com qualidade máxima
        cartoon_final.save(output_path, quality=95, optimize=True)
        print(f"Imagem salva em: {output_path}")
        
        return output_path
    
    def cleanup(self):
        """Libera memória"""
        for model in self.models.values():
            del model
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Instâncias globais
fast_generator = CartoonGANGenerator()
