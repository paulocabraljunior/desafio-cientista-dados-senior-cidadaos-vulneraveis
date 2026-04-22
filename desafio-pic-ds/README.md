# Sistema de Inteligência e Priorização - 1746 (Programa Pequenos Cariocas)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google BigQuery](https://img.shields.io/badge/Google_BigQuery-669DF6?style=for-the-badge&logo=google-cloud&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ffcb1b?style=for-the-badge&logo=lightgbm&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-20232A?style=for-the-badge&logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-black?style=for-the-badge&logo=python&logoColor=white)
![Kepler.gl](https://img.shields.io/badge/Kepler.gl-0A192F?style=for-the-badge&logo=uber&logoColor=white)

## Visão Executiva

Este projeto constitui um pipeline analítico e preditivo escalável desenvolvido para otimizar os recursos de zeladoria urbana e atendimento ao cidadão da Prefeitura do Rio de Janeiro. Ao integrar a base histórica massiva de chamados do portal 1746 (mais de 14M+ registros disponíveis no datario) a APIs climáticas (Open-Meteo) e de calendários (Public Holiday), a solução entrega inteligência acionável. O foco principal não é apenas analisar o passado, mas agir sobre o futuro: garantindo que as equipes de campo sejam direcionadas de forma inteligente para os chamados com maior probabilidade de falha, maximizando o impacto na qualidade de vida da população carioca.

## Arquitetura e Abordagem Metodológica

Como Staff Data Scientist, as decisões de projeto foram pautadas por princípios de MLOps, governança de dados e impacto em políticas públicas:

### Engenharia de Software Aplicada
A solução rejeita o antipadrão de "código espaguete" restrito a notebooks. A lógica pesada de extração de dados (`data_fetcher.py`), tratamento com *scikit-learn pipelines* (`features.py`) e métricas/visualizações (`model_utils.py`) está encapsulada no pacote `src/`. Os notebooks operam estritamente como a camada de apresentação e experimentação orquestrada. Isso garante a manutenibilidade e reduz drasticamente a barreira para a produtização do código.

### Modelagem Preditiva Avançada
Para o problema de classificação ("Resolvido em 7 dias?"), adotou-se o algoritmo **LightGBM**. Suas vantagens de eficiência computacional, processamento nativo de *NaNs* e features categóricas tornam o treinamento altamente iterativo. Além disso, a sintonia de hiperparâmetros (Hyperparameter Tuning) não depende de buscas exaustivas como o *GridSearch*, mas utiliza otimização Bayesiana avançada orientada a grafos com a biblioteca **Optuna**, validada rigorosamente por *Stratified K-Fold* para neutralizar *data leakage*.

### Explicabilidade (XAI) e Governança
Em políticas públicas e gestão estatal, modelos não podem operar como "caixas-pretas". A explicabilidade é um imperativo ético e legal. A solução utiliza **SHAP Values** para gerar visualizações de dependência globais e explicações parciais por instância de chamado. Dessa forma, auditores e gestores públicos podem entender de forma clara quais variáveis (clima, área, dia da semana) influenciaram a previsão de atraso de um chamado específico.

### O Score de Priorização (Business Logic)
Não basta ter um modelo preciso se ele não estiver atrelado aos objetivos estratégicos da prefeitura. O Notebook final introduz um **Score de Prioridade** que combina três pilares:
1. **Eficiência Operacional:** A probabilidade matemática real gerada pelo LightGBM (risco de não resolução em 7 dias).
2. **Equidade Territorial (Vulnerabilidade):** Fatores de correção ponderados aplicando pesos maiores para Áreas de Planejamento (APs) historicamente negligenciadas ou carentes (ex: AP 3 e AP 5).
3. **Impacto de Crise (Clima):** Adicionais táticos baseados no volume pluviométrico diário, elevando a gravidade de podas e alagamentos em dias atípicos.

## Estrutura do Repositório

```text
desafio-pic-ds/
├── README.md
├── requirements.txt
├── src/
│   ├── data_fetcher.py       # Classes para extração de BQ (paginada) e APIs (assíncronas)
│   ├── features.py           # Custom Transformers, Scikit-Learn Pipelines e sampling
│   └── model_utils.py        # Curvas ROC/PR, SHAP Explainability e Lift Curves
├── notebooks/
│   ├── 01_analise_apis_clima.ipynb      # EDA, Correlação cruzada e Análise Geoespacial
│   ├── 02_modelagem_resolucao.ipynb     # Train/Test Split, Modelagem LightGBM, Optuna e XAI
│   └── 03_sistema_priorizacao.ipynb     # Cálculo Híbrido do Score e Simulação com Gain Chart
└── data/                     # Diretório gerado em tempo de execução para os artefatos de teste
```

## Setup e Execução (Reproducibilidade)

Para reproduzir os resultados de ponta a ponta, siga os comandos abaixo.

1. **Clonando e Instalando Dependências:**
```bash
git clone <URL_DO_REPOSITORIO>
cd desafio-pic-ds
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Autenticação GCP (Criticamente Importante):**
Para que a classe `DataFetcher` possa consultar a base do projeto `datario`, você deve se autenticar na nuvem do Google utilizando o Application Default Credentials.
Execute no terminal:
```bash
gcloud auth application-default login
```
*Alternativa:* Configure diretamente o caminho da sua chave de serviço:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/caminho/absoluto/para/sua/chave-de-servico.json"
```

3. **Iniciando a Execução:**
Inicie o ambiente de notebooks. Garanta que você execute a ordem estrita (`01 -> 02 -> 03`), visto que o fluxo cria artefatos preditivos (`../data/test_predictions.csv`) necessários para simular as regras do negócio na etapa de priorização.
```bash
jupyter notebook
```

## Conclusões e Próximos Passos (Visão MLOps Produtiva)
Para escalar esta prova de conceito para um ambiente crítico 24/7 da Prefeitura (e.g. no GCP), os próximos passos evolutivos compreendem:
- **Orquestração:** Migrar as rotinas de extração e a predição em lote do LightGBM para DAGs do **Apache Airflow** ou **Prefect**.
- **Model Registry & Tracking:** Implementar **MLflow** para versionar os artefatos serializados dos modelos, hiperparâmetros experimentados e métricas (Precision, Recall, ROC AUC) garantindo rastreabilidade histórica completa.
- **Monitoramento de Data Drift:** Acoplar soluções como **Evidently AI** nos fluxos preditivos para emitir alertas caso haja mudanças severas no comportamento do cidadão (covariate shift) ou quebras nos padrões sazonais identificados pelo sistema.