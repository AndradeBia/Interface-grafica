# HEVA: Pipeline Integrado para Análise de Lesões de Pele 🔬

Bem-vindo ao repositório do **HEVA** (Hybrid Ensemble for Vision Analysis) — um projeto que implementa uma interface gráfica para um **pipeline robusto de machine learning em duas etapas** voltado à classificação de lesões de pele.

A aplicação foi construída com **Gradio** e permite que o usuário faça upload de uma imagem de qualquer tamanho, que será processada da seguinte forma:

1. **Segmentação Semântica (Segformer):** Identificação e isolamento da lesão.  
2. **Classificação em Ensemble:** Combinação de descritores clássicos e redes neurais profundas para classificar a lesão como **Benigna** ou **Maligna**.

---

## ✨ Principais Funcionalidades

- **Interface Amigável:** Webapp simples e intuitivo em Gradio.  
- **Pipeline em Duas Etapas:** Segmentação + Classificação, simulando o processo de análise de um especialista.  
- **Ensemble Híbrido:** Combina descritores clássicos (**LBP**) com redes neurais (**ResNet50 e ViT**).  
- **Flexibilidade:** Aceita imagens de diferentes tamanhos e resoluções.  
- **Feedback Visual:** Exibe a máscara de segmentação junto ao resultado final.  

---

## 🏗️ Arquitetura do Modelo

O pipeline é dividido em duas grandes etapas:

### 🔹 Etapa 1 — Segmentação com Segformer
- Utiliza o modelo **Segformer** (`nvidia/segformer-b5-finetuned-ade-640-640`) ajustado para detectar lesões de pele.  
- Gera uma **máscara binária**, isolando a região de interesse.  

### 🔹 Etapa 2 — Classificação com Ensemble Híbrido
Após a segmentação, a área da lesão é recortada e processada para extração de **três tipos de características**:

1. **Descritores de Textura (LBP):** Capturam padrões da superfície.  
2. **Features da ResNet50:** Extração hierárquica de representações.  
3. **Features do Vision Transformer (ViT):** Relações globais e contextuais (`google/vit-base-patch16-224-in21k`).  

Essas features são concatenadas e classificadas por um **SVM** (Support Vector Machine).  

---

## 🚀 Instalação e Execução

### 📌 Pré-requisitos
- Python **3.8+**  
- Git  
- **Git LFS** (necessário para os arquivos grandes dos modelos)  

### 1. Instalar o Git LFS
Baixe e instale o Git LFS em: [git-lfs.github.com](https://git-lfs.github.com)  
Depois, configure no terminal:  
```bash
git lfs install
```

### 2. Clonar o Repositório
```bash
git clone https://github.com/AndradeBia/Interface-grafica.git
cd Interface-grafica
```
> ⚠️ Caso os modelos não sejam baixados corretamente (aparecendo como arquivos pequenos de texto), execute:  
```bash
git lfs pull
```

### 3. Criar Ambiente Virtual
```bash
# Criar
python -m venv venv

# Ativar (Windows)
.\venv\Scripts\activate

# Ativar (Linux/macOS)
source venv/bin/activate
```

### 4. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 5. Executar a Aplicação
```bash
python app.py
```
Aguarde a mensagem **"🧠 Carregando todos os modelos..."**.  
Depois, acesse a aplicação no navegador em [http://127.0.0.1:7860](http://127.0.0.1:7860).  

---

## 📁 Estrutura do Projeto
```
└── Interface-grafica/
    ├── app.py                      # Código principal da aplicação Gradio e pipeline
    ├── requirements.txt            # Dependências do projeto
    ├── feature_extractor_finetuned.h5   # Modelo ResNet50 para extração de features
    ├── segformer_best_model.pth    # Pesos do Segformer treinado
    ├── svm_pipeline_artifacts.pkl  # Artefatos do SVM (modelo, scaler, etc.)
    └── vit_model/                  # Pasta do modelo ViT
        ├── config.json
        └── model.safetensors
```
---

💡 **HEVA** combina **deep learning** e **descritores clássicos** para uma análise precisa de lesões cutâneas, com foco em usabilidade e transparência no processo.

