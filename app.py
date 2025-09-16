# app.py

import gradio as gr
import numpy as np
import pickle
import warnings
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis, entropy
from PIL import Image
import cv2 # Usaremos OpenCV para BBox, Resize e Filtro Gaussiano

# ===================================================================
# CONFIGURAÇÃO E CARREGAMENTO DOS MODELOS
# ===================================================================

# Silencia avisos
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

DEVICE_TORCH = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Funções de Extração de Features (do pipeline de CLASSIFICAÇÃO) ---
def extract_features(image, mask):
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool): return {}
    coords = np.argwhere(mask_bool)
    r0, c0 = coords.min(axis=0); r1, c1 = coords.max(axis=0) + 1
    image_crop = image[r0:r1, c0:c1]; mask_crop = mask_bool[r0:r1, c0:c1]
    
    if image_crop.size == 0 or not np.any(mask_crop): return {}
    
    gray_crop = (cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)).astype(np.uint8)
    median_val = np.median(gray_crop[mask_crop])
    gray_crop_filled = gray_crop.copy(); gray_crop_filled[~mask_crop] = median_val
    feats = {}
    radius, n_points = 1, 8
    lbp = local_binary_pattern(gray_crop_filled, n_points, radius, 'uniform')
    n_bins = n_points + 2
    hist,_ = np.histogram(lbp[mask_crop].ravel(), bins=n_bins, range=(0, n_bins)); hist = hist.astype(float); hist /= (hist.sum()+1e-6)
    for i,v in enumerate(hist): feats[f'lbp_bin_{i}'] = v
    return feats

def extract_resnet_features_inference(model, imgs):
    imgs_float = imgs.astype(np.float32)
    preprocessed_imgs = tf.keras.applications.resnet50.preprocess_input(imgs_float)
    return model.predict(preprocessed_imgs, verbose=0)

def preprocess_vit(img):
    img_resized = cv2.resize(img, (224, 224))
    t = torch.from_numpy(img_resized).permute(2,0,1).float()
    return (t/255.0-0.5)*2

def extract_cls_inference(model, imgs, device):
    with torch.no_grad():
        batch_imgs = torch.stack([preprocess_vit(img) for img in imgs]).to(device)
        hs = model(batch_imgs, output_hidden_states=True).hidden_states[-1]
    return hs[:,0,:].cpu().numpy()


print("🧠 Carregando todos os modelos e artefatos... Por favor, aguarde.")
try:
    # --- 1. Carrega artefatos do SVM (CLASSIFICAÇÃO) ---
    with open("svm_pipeline_artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    SCALER = art["scaler"]
    BEST_KERNEL = art["best_kernel"]
    SVM_MODEL = art["svm_models"][BEST_KERNEL]
    FEAT_NAMES = art["feature_names"]

    # --- 2. Carrega extratores de features de Deep Learning (CLASSIFICAÇÃO) ---
    RESNET_MODEL = tf.keras.models.load_model("feature_extractor_finetuned.h5", compile=False)
    VIT_MODEL = ViTForImageClassification.from_pretrained("./vit_model")
    VIT_MODEL.config.output_hidden_states = True
    VIT_MODEL.eval().to(DEVICE_TORCH)
    
    # --- 3. NOVO: Carrega o modelo Segformer (SEGMENTAÇÃO) ---
    MODEL_NAME_SEG = "nvidia/segformer-b5-finetuned-ade-640-640"
    MODEL_PATH_SEG = "segformer_best_model.pth"
    ID2LABEL_SEG = {0: "background", 1: "lesion"}
    
    feature_extractor_seg = SegformerFeatureExtractor.from_pretrained(MODEL_NAME_SEG)
    model_seg = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME_SEG,
        num_labels=len(ID2LABEL_SEG),
        id2label=ID2LABEL_SEG,
        ignore_mismatched_sizes=True # <-- CORREÇÃO APLICADA AQUI
    )
    model_seg.load_state_dict(torch.load(MODEL_PATH_SEG, map_location=DEVICE_TORCH))
    model_seg.to(DEVICE_TORCH)
    model_seg.eval()
    
    CLASS_MAP = {0: "Benigno", 1: "Maligno"}
    
    print(f"✅ Modelos carregados com sucesso! Usando dispositivo: {DEVICE_TORCH}")

except FileNotFoundError as e:
    print(f"❌ Erro Crítico: Não foi possível carregar o arquivo '{e.filename}'. Verifique se todos os arquivos estão no diretório.")
    exit()

# ===================================================================
# NOVAS FUNÇÕES HELPER PARA O PIPELINE INTEGRADO
# ===================================================================

def infer_mask_from_image(model, feature_extractor, image_np, device):
    """Usa o Segformer para gerar uma máscara a partir de uma imagem de qualquer tamanho."""
    image_pil = Image.fromarray(image_np).convert("RGB")
    
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
    
    upsampled_logits = nn.functional.interpolate(
        logits, size=image_pil.size[::-1], mode='bilinear', align_corners=False
    )
    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred_mask

def preprocess_for_feature_extractors(image, mask):
    """Executa o pipeline de BBox, Crop, Resize e Filtro Gaussiano."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    image_cropped = image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    
    image_resized = cv2.resize(image_cropped, (100, 100), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_cropped, (100, 100), interpolation=cv2.INTER_NEAREST)
    
    image_blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    
    return image_blurred, mask_resized

# ===================================================================
# FUNÇÃO DE PREDIÇÃO (O "Backend" do Gradio) - ADAPTADA
# ===================================================================

def predict_pipeline_integrado(input_image):
    """
    Recebe uma imagem crua de qualquer tamanho, executa todo o pipeline 
    e retorna a máscara gerada e a classificação final.
    """
    if input_image is None:
        return "Por favor, forneça uma imagem.", None

    # --- PASSO 1: Inferir a máscara com o Segformer ---
    binary_mask = infer_mask_from_image(model_seg, feature_extractor_seg, input_image, DEVICE_TORCH)
    
    if np.sum(binary_mask) == 0:
        return "Nenhuma lesão detectada pelo modelo de segmentação.", None
    
    # --- PASSO 2: Pré-processar a imagem e a máscara para os extratores ---
    image_100, mask_100 = preprocess_for_feature_extractors(input_image, binary_mask)

    if image_100 is None:
        return "Erro ao processar a máscara gerada.", None

    # --- PASSO 3: Pipeline de Classificação Original ---
    img_batch = np.expand_dims(image_100, axis=0)
    
    fdict = extract_features(image_100, mask_100)
    desc_new = np.array([[fdict.get(k, 0.0) for k in FEAT_NAMES]])
    
    res_new = extract_resnet_features_inference(RESNET_MODEL, img_batch)
    vit_new = extract_cls_inference(VIT_MODEL, img_batch, device=DEVICE_TORCH)
    
    # --- 4. Combinação, Normalização e Predição Final ---
    all_new = np.hstack((desc_new, vit_new, res_new))
    all_new_scaled = SCALER.transform(all_new)
    
    prediction_idx = SVM_MODEL.predict(all_new_scaled)[0]
    prediction_label = CLASS_MAP.get(prediction_idx, "Classe Desconhecida")
    
    # Prepara a máscara para visualização (converte de binária para RGB)
    mask_visual = (binary_mask * 255).astype(np.uint8)
    mask_visual_rgb = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2RGB)

    return {prediction_label: 1.0}, mask_visual_rgb

# ===================================================================
# DEFINIÇÃO DA INTERFACE GRADIO - ADAPTADA
# ===================================================================

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 🔬 Demonstrador do Pipeline Integrado de Análise de Lesões de Pele
        Faça o upload de uma imagem de lesão de pele de **qualquer tamanho**.
        1. O modelo **Segformer** irá primeiro gerar a máscara de segmentação.
        2. A imagem será pré-processada (recorte, redimensionamento, filtro).
        3. O pipeline de **Classificação** (SVM + ResNet + ViT) dará o diagnóstico final.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Imagem de Entrada (Tamanho Variado, RGB)")
            predict_btn = gr.Button("Executar Análise Completa", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                output_mask = gr.Image(type="numpy", label="Máscara Gerada pelo Segformer")
                output_label = gr.Label(label="Resultado da Classificação")
            
    predict_btn.click(
        fn=predict_pipeline_integrado,
        inputs=input_image,
        outputs=[output_label, output_mask]
    )

if __name__ == "__main__":
    iface.launch()
