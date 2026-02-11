import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import CannyDetector
from transformers import pipeline as transformers_pipeline
from PIL import Image
import numpy as np
import cv2
import os
import json
import time
from torchvision import transforms


def setup_local_cache():
    """Configura cache local do Hugging Face na pasta do projeto"""
    # Obter diretório do projeto
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(project_dir, "huggingface")

    # Criar diretório se não existir
    os.makedirs(cache_dir, exist_ok=True)

    # Configurar variáveis de ambiente para HF cache
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir

    print(f"Cache configurado para: {cache_dir}")
    return cache_dir


def load_styles_config():
    """Carrega configuração de estilos do arquivo JSON"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    styles_path = os.path.join(project_dir, "styles.json")

    with open(styles_path, "r", encoding="utf-8") as f:
        return json.load(f)


class CartoonGenerator:
    def __init__(self):
        # Configurar cache local antes de qualquer carregamento
        setup_local_cache()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.controlnet_pipe = None
        self.img2img_pipe = None
        self.depth_pipe = None
        self.pix2pix_pipe = None
        self.gfpgan_model = None
        self.canny_detector = None
        self.depth_estimator = None
        print(f"Usando dispositivo: {self.device}")

    def load_model(self, method="controlnet"):
        """
        Carrega os modelos Stable Diffusion

        Args:
            method: "controlnet", "img2img", "depth", "pix2pix", ou "gfpgan"
        """
        if method == "controlnet":
            if self.controlnet_pipe is not None:
                return

            print(
                "Carregando ControlNet... Isso pode demorar alguns minutos na primeira vez."
            )

            # Carregar ControlNet Canny
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Carregar Stable Diffusion com ControlNet
            self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )

            # Otimizações
            self.controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.controlnet_pipe.scheduler.config
            )
            self.controlnet_pipe = self.controlnet_pipe.to(self.device)

            # Habilitar otimizações de memória
            if self.device == "cuda":
                self.controlnet_pipe.enable_model_cpu_offload()

            # Detector Canny
            self.canny_detector = CannyDetector()

            print("ControlNet carregado com sucesso!")

        elif method == "img2img":
            if self.img2img_pipe is not None:
                return

            print(
                "Carregando Img2Img... Isso pode demorar alguns minutos na primeira vez."
            )

            # Carregar Stable Diffusion Img2Img
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )

            # Otimizações
            self.img2img_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.img2img_pipe.scheduler.config
            )
            self.img2img_pipe = self.img2img_pipe.to(self.device)

            # Habilitar otimizações de memória
            if self.device == "cuda":
                self.img2img_pipe.enable_model_cpu_offload()

            print("Img2Img carregado com sucesso!")
            
        elif method == "depth":
            if self.depth_pipe is not None:
                return

            print(
                "Carregando ControlNet Depth... Isso pode demorar alguns minutos na primeira vez."
            )

            # Carregar ControlNet Depth
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Carregar Stable Diffusion com ControlNet Depth
            self.depth_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )

            # Otimizações
            self.depth_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.depth_pipe.scheduler.config
            )
            self.depth_pipe = self.depth_pipe.to(self.device)

            # Habilitar otimizações de memória
            if self.device == "cuda":
                self.depth_pipe.enable_model_cpu_offload()

            # Estimador de profundidade
            self.depth_estimator = transformers_pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device == "cuda" else -1
            )

            print("ControlNet Depth carregado com sucesso!")
            
        elif method == "pix2pix":
            if self.pix2pix_pipe is not None:
                return

            print(
                "Carregando Instruct-Pix2Pix... Isso pode demorar alguns minutos na primeira vez."
            )

            # Carregar Instruct-Pix2Pix
            self.pix2pix_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            )

            self.pix2pix_pipe = self.pix2pix_pipe.to(self.device)

            # Habilitar otimizações de memória
            if self.device == "cuda":
                self.pix2pix_pipe.enable_model_cpu_offload()

            print("Instruct-Pix2Pix carregado com sucesso!")
            
        elif method == "gfpgan":
            if self.gfpgan_model is not None and self.img2img_pipe is not None:
                return

            print(
                "Carregando GFPGAN + Img2Img... Isso pode demorar alguns minutos na primeira vez."
            )

            # Carregar GFPGAN para restauração facial
            try:
                from gfpgan import GFPGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                model_path = os.path.join(
                    os.path.dirname(__file__),
                    "huggingface",
                    "GFPGANv1.4.pth"
                )
                
                # Baixar modelo se não existir
                if not os.path.exists(model_path):
                    import urllib.request
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    print("Baixando modelo GFPGAN...")
                    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                    urllib.request.urlretrieve(url, model_path)
                
                self.gfpgan_model = GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
            except Exception as e:
                print(f"Aviso: GFPGAN não disponível ({e}). Usando apenas Img2Img.")
                self.gfpgan_model = None

            # Carregar Img2Img também
            if self.img2img_pipe is None:
                self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                )
                self.img2img_pipe.scheduler = UniPCMultistepScheduler.from_config(
                    self.img2img_pipe.scheduler.config
                )
                self.img2img_pipe = self.img2img_pipe.to(self.device)
                if self.device == "cuda":
                    self.img2img_pipe.enable_model_cpu_offload()

            print("GFPGAN + Img2Img carregado com sucesso!")

    def process_image(
        self, image_path, output_path, style="cartoon", method="controlnet"
    ):
        """
        Processa a imagem e transforma em cartoon

        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho para salvar a imagem gerada
            style: Estilo do cartoon (cartoon, anime, comic, etc)
            method: "controlnet", "img2img", "depth", "pix2pix", ou "gfpgan"

        Returns:
            tuple: (output_path, processing_time)
        """
        start_time = time.time()

        self.load_model(method)

        # Carregar e preparar imagem
        input_image = Image.open(image_path).convert("RGB")

        # Redimensionar para melhor performance (máximo 768px)
        max_size = 768
        ratio = min(max_size / input_image.width, max_size / input_image.height)
        if ratio < 1:
            new_size = (int(input_image.width * ratio), int(input_image.height * ratio))
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)

        # Carregar prompts do arquivo JSON
        styles_config = load_styles_config()
        quality_styles = styles_config.get("quality", {})

        # Obter configuração do estilo
        style_config = quality_styles.get(style, quality_styles.get("cartoon", {}))
        prompt = style_config.get(
            "prompt", "cartoon style, vibrant colors, detailed, high quality"
        )
        negative_prompt = "ugly, blurry, low quality, distorted, deformed, bad anatomy, watermark, text"

        print(f"Gerando imagem no estilo {style} usando {method}...")

        # Gerar imagem baseado no método
        with torch.inference_mode():
            if method == "controlnet":
                # Detectar bordas com Canny
                print("Detectando bordas...")
                canny_image = self.canny_detector(input_image)

                result = self.controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=canny_image,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=0.8,
                ).images[0]
                
            elif method == "depth":
                # Estimar mapa de profundidade
                print("Estimando mapa de profundidade...")
                depth_map = self.depth_estimator(input_image)["depth"]
                
                # Converter para PIL Image
                import numpy as np
                depth_array = depth_map.cpu().numpy()
                depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255
                depth_image = Image.fromarray(depth_array.astype(np.uint8)).convert("RGB")
                
                result = self.depth_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=depth_image,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=0.8,
                ).images[0]
                
            elif method == "pix2pix":
                # Instruct-Pix2Pix usa instruções para transformação
                instruction = f"Transform into {prompt}"
                print(f"Aplicando instrução: {instruction}")
                
                result = self.pix2pix_pipe(
                    instruction,
                    image=input_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7.5,
                ).images[0]
                
            elif method == "gfpgan":
                # Primeiro aplica GFPGAN para restaurar o rosto
                print("Restaurando rosto com GFPGAN...")
                import cv2
                import numpy as np
                
                # Converter PIL para numpy/cv2
                img_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
                
                # Aplicar GFPGAN
                _, _, output = self.gfpgan_model.enhance(
                    img_cv, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                
                # Converter de volta para PIL
                restored_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                
                # Depois aplica img2img para estilizar
                print("Aplicando estilo cartoon...")
                result = self.img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=restored_image,
                    strength=0.55,  # Menor strength para preservar restauração
                    num_inference_steps=25,
                    guidance_scale=7.0,
                ).images[0]
                
            else:  # img2img (default)
                result = self.img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=0.65,  # Preserva 35% da imagem original
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]

        # Salvar resultado
        result.save(output_path)

        # Calcular tempo total
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Imagem salva em: {output_path}")
        print(f"⏱️ Tempo de processamento: {processing_time:.2f} segundos")

        return output_path, processing_time

    def cleanup(self):
        """Libera memória"""
        if self.controlnet_pipe is not None:
            del self.controlnet_pipe
            self.controlnet_pipe = None
        if self.img2img_pipe is not None:
            del self.img2img_pipe
            self.img2img_pipe = None
        if self.depth_pipe is not None:
            del self.depth_pipe
            self.depth_pipe = None
        if self.pix2pix_pipe is not None:
            del self.pix2pix_pipe
            self.pix2pix_pipe = None
        if self.gfpgan_model is not None:
            del self.gfpgan_model
            self.gfpgan_model = None
        if self.depth_estimator is not None:
            del self.depth_estimator
            self.depth_estimator = None
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
                "bryandlee/animegan2-pytorch:main",
                "generator",
                pretrained=style,
                device=self.device,
                progress=True,
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

        Returns:
            tuple: (output_path, processing_time)
        """
        start_time = time.time()

        model = self.load_model(style)

        # Carregar imagem
        img = Image.open(image_path).convert("RGB")

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
        square_img = Image.new("RGB", (max_size, max_size), (255, 255, 255))
        offset_x = (max_size - new_width) // 2
        offset_y = (max_size - new_height) // 2
        square_img.paste(img_resized, (offset_x, offset_y))

        # Preparar transformações com normalização adequada
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

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
        cartoon_cropped = cartoon_img.crop(
            (offset_x, offset_y, offset_x + new_width, offset_y + new_height)
        )

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

        # Calcular tempo total
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Imagem salva em: {output_path}")
        print(f"⏱️ Tempo de processamento: {processing_time:.2f} segundos")

        return output_path, processing_time

    def cleanup(self):
        """Libera memória"""
        for model in self.models.values():
            del model
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Instâncias globais
fast_generator = CartoonGANGenerator()
