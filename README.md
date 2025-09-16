
````markdown
# HEVA: Pipeline Integrado para Análise de Lesões de Pele 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bem-vindo ao repositório do **HEVA** (Hybrid Ensemble for Vision Analysis), um projeto que implementa uma interface gráfica para um robusto pipeline de machine learning de duas etapas para a classificação de lesões de pele.

A aplicação, construída com **Gradio**, permite que o usuário faça o upload de uma imagem de qualquer tamanho. A imagem então passa por:
1.  Um modelo de **segmentação semântica (Segformer)** para identificar e isolar a lesão.
2.  Um poderoso **modelo de classificação em ensemble** que combina descritores de textura, features da ResNet e do Vision Transformer (ViT) para classificar a lesão como "Benigna" ou "Maligna".



## ✨ Principais Funcionalidades

-   **Interface Amigável:** Interface web simples e intuitiva criada com Gradio.
-   **Pipeline de Duas Etapas:** Primeiro segmenta, depois classifica, imitando o foco de um especialista.
-   **Modelo Ensemble Híbrido:** Combina a força dos descritores clássicos (LBP) com o poder de representação de modelos de deep learning (ResNet e ViT).
-   **Flexibilidade de Entrada:** Aceita imagens de lesões de pele de diferentes tamanhos e resoluções.
-   **Feedback Visual:** Além da classificação, a interface exibe a máscara de segmentação gerada pelo modelo, mostrando qual área da imagem foi analisada.

## 🏗️ Arquitetura do Modelo

O pipeline do HEVA é dividido em duas etapas principais:

### Etapa 1: Segmentação Semântica com Segformer

Qualquer imagem de entrada é primeiramente processada por um modelo **Segformer** (especificamente, `nvidia/segformer-b5-finetuned-ade-640-640`) que foi afinado para identificar lesões de pele. O resultado é uma máscara binária que isola a região de interesse.

### Etapa 2: Classificação com Ensemble Híbrido

Após o isolamento da lesão, a área é recortada e pré-processada. Em seguida, extraímos três conjuntos de características distintas:

1.  **Descritores de Textura:** Através do **Local Binary Pattern (LBP)**, capturamos características de textura da superfície da lesão.
2.  **Features da ResNet50:** Utilizamos uma **ResNet50** pré-treinada para extrair features hierárquicas da imagem.
3.  **Features do Vision Transformer (ViT):** Usamos um **ViT** (`google/vit-base-patch16-224-in21k`) para capturar relações globais e contextuais na imagem da lesão.

Finalmente, essas três fontes de informação são concatenadas e alimentam um classificador **Support Vector Machine (SVM)**, que realiza o diagnóstico final.

## 🚀 Instalação e Execução

Para executar esta aplicação localmente, siga os passos abaixo.

### Pré-requisitos

-   Python 3.8+
-   Git
-   **Git LFS** (Large File Storage)

### 1. Instalação do Git LFS

Este repositório utiliza o Git LFS para gerenciar os arquivos grandes dos modelos. **É crucial que você instale o Git LFS** no seu computador antes de clonar o repositório.

-   Visite [git-lfs.github.com](https://git-lfs.github.com) e siga as instruções de download e instalação para o seu sistema operacional.
-   Após a instalação, configure o Git LFS executando o seguinte comando no seu terminal:
    ```bash
    git lfs install
    ```

### 2. Clonando o Repositório

Com o Git LFS instalado, clone o repositório. Os arquivos grandes dos modelos serão baixados automaticamente durante o processo de `clone`.

```bash
git clone [https://github.com/AndradeBia/Interface-grafica.git](https://github.com/AndradeBia/Interface-grafica.git)
cd Interface-grafica
```
*Se os arquivos dos modelos não forem baixados (e aparecerem como pequenos arquivos de texto), execute `git lfs pull` dentro da pasta do projeto.*

### 3. Configurando o Ambiente Virtual

É uma boa prática criar um ambiente virtual para isolar as dependências do projeto.

```bash
# Criar um ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
.\venv\Scripts\activate

# Ativar o ambiente (macOS/Linux)
source venv/bin/activate
```

### 4. Instalando as Dependências

Instale todas as bibliotecas necessárias com um único comando:

```bash
pip install -r requirements.txt
```

### 5. Executando a Aplicação

Com tudo instalado, inicie a interface Gradio:

```bash
python app.py
```
Aguarde a mensagem "🧠 Carregando todos os modelos...", que pode levar alguns instantes. Após o carregamento, acesse o endereço local (geralmente `http://127.0.0.1:7860`) que aparecerá no seu terminal.

## 📁 Estrutura dos Arquivos

```
└── andradebia-interface-grafica/
    ├── app.py                      # Código principal da aplicação Gradio e do pipeline.
    ├── requirements.txt            # Lista de dependências Python.
    ├── feature_extractor_finetuned.h5 # Modelo ResNet50 para extração de features.
    ├── segformer_best_model.pth    # Pesos do modelo Segformer afinado.
    ├── svm_pipeline_artifacts.pkl  # Artefatos do SVM (modelo, scaler, etc.).
    └── vit_model/                    # Pasta contendo o modelo ViT.
        ├── config.json
        └── model.safetensors
```

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
````
