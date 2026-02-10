from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import json
from werkzeug.utils import secure_filename
from cartoon_models import generator, fast_generator
import uuid
from datetime import datetime

app = Flask(__name__)

# Configurações
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["OUTPUT_FOLDER"] = "static/outputs"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}

# Criar pastas se não existirem
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def allowed_file(filename):
    """Verifica se o arquivo tem extensão permitida"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    """Página principal"""
    return render_template("index.html")


@app.route("/api/styles")
def get_styles():
    """Retorna os estilos disponíveis do arquivo JSON"""
    try:
        styles_path = os.path.join(os.path.dirname(__file__), "styles.json")
        with open(styles_path, 'r', encoding='utf-8') as f:
            styles = json.load(f)
        return jsonify(styles)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    """Processa o upload e gera o cartoon"""
    try:
        # Verificar se há arquivo
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {"error": "Tipo de arquivo não permitido. Use PNG, JPG ou JPEG"}
                ),
                400,
            )

        # Pegar parâmetros
        style = request.form.get("style", "cartoon")
        mode = request.form.get("mode", "fast")  # fast ou quality

        # Gerar nome único para o arquivo
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salvar arquivo original
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit(".", 1)[1].lower()
        input_filename = f"{timestamp}_{unique_id}_input.{file_ext}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
        file.save(input_path)

        # Gerar cartoon
        output_filename = f"{timestamp}_{unique_id}_cartoon.png"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

        print(f"Processando imagem: {input_path} | Modo: {mode} | Estilo: {style}")
        
        # Escolher gerador baseado no modo
        if mode == "fast":
            fast_generator.process_image(input_path, output_path, style=style)
        else:  # quality
            generator.process_image(input_path, output_path, style=style)

        return jsonify(
            {
                "success": True,
                "input_image": f"/static/uploads/{input_filename}",
                "output_image": f"/static/outputs/{output_filename}",
            }
        )

    except Exception as e:
        print(f"Erro ao processar imagem: {str(e)}")
        return jsonify({"error": f"Erro ao processar imagem: {str(e)}"}), 500


@app.route("/static/<path:path>")
def send_static(path):
    """Serve arquivos estáticos"""
    return send_from_directory("static", path)


if __name__ == "__main__":
    print("=" * 60)
    print("🎨 Gerador de Cartoon - Modo Rápido e Qualidade")
    print("=" * 60)
    print("\nCarregando aplicação...")
    print("\n⚠️  IMPORTANTE:")
    print("- MODO RÁPIDO (CartoonGAN): ~100MB, 1-5 segundos")
    print("- MODO QUALIDADE (ControlNet): ~6GB, 30-60 segundos")
    print("- Os modelos serão baixados automaticamente na primeira vez")
    print("- GPU é recomendada para melhor performance\n")
    print("=" * 60)

    app.run(debug=True, host="0.0.0.0", port=5000)
