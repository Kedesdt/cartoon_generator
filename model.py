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


class CartoonGenerator:
    def __init__(self):
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        print(f"CartoonGAN usando dispositivo: {self.device}")
    
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
        
        # Preparar transformações
        face2paint = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        # Carregar imagem
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Converter para tensor
        img_tensor = face2paint(img).unsqueeze(0)
        
        # Mover para device
        img_tensor = img_tensor.to(self.device)
        
        print(f"Gerando cartoon no estilo {style}...")
        
        # Gerar cartoon
        with torch.no_grad():
            cartoon_tensor = model(img_tensor)
        
        # Converter de volta para imagem
        cartoon_tensor = cartoon_tensor.cpu().squeeze(0)
        cartoon_img = transforms.ToPILImage()(cartoon_tensor)
        
        # Redimensionar de volta ao tamanho original
        cartoon_img = cartoon_img.resize(original_size, Image.Resampling.LANCZOS)
        
        # Salvar
        cartoon_img.save(output_path)
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
