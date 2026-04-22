# Desafio Técnico - Cientista de Dados Sênior (Pequenos Cariocas 1746)

Este repositório contém a solução proposta para o desafio técnico de Staff/Senior Data Scientist para a Prefeitura do Rio de Janeiro.

## Estrutura do Projeto

```
desafio-pic-ds/
├── README.md
├── requirements.txt
├── src/
│   ├── data_fetcher.py       # Extração de dados do BigQuery e APIs (Assíncrono)
│   ├── features.py           # Pipelines do Scikit-Learn e amostragem
│   └── model_utils.py        # Utilitários de gráficos (ROC, PR, SHAP, Lift)
└── notebooks/
    ├── 01_analise_apis_clima.ipynb     # EDA, espacial e Time Series
    ├── 02_modelagem_resolucao.ipynb    # Pipeline ML, LightGBM, Optuna, SHAP
    └── 03_sistema_priorizacao.ipynb    # Score de Prioridade e Lift Curve
```

## Como Executar

1. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Certifique-se de ter suas credenciais do Google Cloud configuradas (se for acessar o BigQuery de produção):
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/chave.json"
```
*(Nota: O código possui fallback interno para gerar dados sintéticos em caso de falha de autenticação)*

3. Inicie o Jupyter e navegue pelos notebooks na ordem.
```bash
jupyter notebook
```

## Destaques da Solução (Nível Sênior)
- **Engenharia de Dados Realista**: Queries otimizadas no BigQuery com limites e filtro de partição. Consultas assíncronas em APIs REST usando `aiohttp`.
- **Prevenção de Data Leakage**: Validação temporal estrita (Treino no passado, Teste no futuro).
- **Tratamento Avançado de ML**: Pipelines com `ColumnTransformer` do `scikit-learn`, otimização bayesiana com `Optuna`.
- **Explainable AI (XAI)**: Uso avançado de gráficos de dependência do `SHAP` para justificar previsões aos stakeholders.
- **Métricas Focadas em Negócio**: Análise com curvas de Precision-Recall e cálculo do Gain Chart (Lift) priorizando métricas atreladas à gestão pública (não perder casos críticos).
