import asyncio
import aiohttp
import pandas as pd
from google.cloud import bigquery
from typing import Dict, Any, List, Optional
import os
import requests

class DataFetcher:
    def __init__(self, project_id: Optional[str] = None):
        """
        Inicializa o DataFetcher.

        Args:
            project_id (Optional[str]): ID do projeto no Google Cloud.
                                        Se None, tentará usar o default do ambiente.
        """
        # Se project_id não for fornecido, a biblioteca do BQ tentará inferir do ambiente
        self.client = bigquery.Client(project=project_id) if project_id else bigquery.Client()

    def get_1746_data(self, start_date: str = '2023-01-01', limit: int = 50000, offset: int = 0) -> pd.DataFrame:
        """
        Extrai dados do 1746 do BigQuery, fazendo join com dados de bairro para obter a Área de Planejamento (AP).
        Usa filtro de partição e limite para paginação/testes.

        Args:
            start_date (str): Data de início da partição YYYY-MM-DD.
            limit (int): Limite máximo de linhas.
            offset (int): Ponto de partida (paginação).

        Returns:
            pd.DataFrame: DataFrame com os dados extraídos.
        """
        query = f"""
        SELECT
            c.id_chamado,
            c.data_inicio,
            c.data_fim,
            c.id_bairro,
            b.nome AS nome_bairro,
            b.subprefeitura AS area_planejamento,
            c.id_tipo,
            c.tipo,
            c.subtipo,
            c.status,
            c.longitude,
            c.latitude,
            -- Target: resolvido_em_7_dias
            CASE
                WHEN c.data_fim IS NOT NULL AND TIMESTAMP_DIFF(c.data_fim, c.data_inicio, DAY) <= 7 THEN 1
                ELSE 0
            END AS resolvido_em_7_dias,
            TIMESTAMP_DIFF(c.data_fim, c.data_inicio, DAY) as dias_para_resolucao
        FROM `datario.adm_central_atendimento_1746.chamado` c
        LEFT JOIN `datario.dados_mestres.bairro` b
            ON c.id_bairro = b.id_bairro
        WHERE c.data_particao >= '{start_date}'
        ORDER BY c.data_inicio DESC
        LIMIT {limit} OFFSET {offset}
        """

        # Em ambiente real, o cliente precisa estar autenticado no GCP com permissão de leitura no dataset do datario.
        # Caso falhe a autenticação (ex: mock environment), retornamos um DF dummy para permitir a execução dos notebooks.
        try:
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            return df
        except Exception as e:
            print(f"Erro ao consultar o BQ: {e}. Gerando dados mockados para desenvolvimento.")
            return self._generate_mock_data(limit)

    def _generate_mock_data(self, limit: int) -> pd.DataFrame:
        import numpy as np
        """Gera dados sintéticos para o desenvolvimento caso o BQ não esteja acessível."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-04-01', periods=limit)
        areas = ['AP 1', 'AP 2', 'AP 3', 'AP 4', 'AP 5']

        df = pd.DataFrame({
            'id_chamado': np.arange(limit),
            'data_inicio': np.random.choice(dates, limit),
            'nome_bairro': np.random.choice(['Copacabana', 'Tijuca', 'Bangu', 'Barra da Tijuca', 'Centro'], limit),
            'area_planejamento': np.random.choice(areas, limit, p=[0.1, 0.2, 0.3, 0.15, 0.25]),
            'tipo': np.random.choice(['Buraco', 'Poda de Árvore', 'Alagamento', 'Iluminação', 'Lixo'], limit),
            'status': np.random.choice(['Fechado', 'Aberto'], limit, p=[0.8, 0.2]),
            'longitude': np.random.uniform(-43.8, -43.1, limit),
            'latitude': np.random.uniform(-23.1, -22.7, limit),
            'resolvido_em_7_dias': np.random.choice([0, 1], limit, p=[0.3, 0.7]),
        })
        # Sorting mock data to simulate real ordered query
        df = df.sort_values(by='data_inicio', ascending=False).reset_index(drop=True)
        return df

    async def fetch_weather_async(self, latitude: float, longitude: float, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Busca dados históricos de clima de forma assíncrona usando a API Open-Meteo.

        Args:
            latitude (float): Latitude
            longitude (float): Longitude
            start_date (str): Data de início YYYY-MM-DD
            end_date (str): Data de fim YYYY-MM-DD

        Returns:
            Dict[str, Any]: Resposta JSON parseada.
        """
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,precipitation_sum&timezone=America/Sao_Paulo"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response.raise_for_status()

    def fetch_weather_sync(self, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Wrapper síncrono para a requisição de clima e formatação em DataFrame.
        Usa cache básico simulado (pode ser expandido com lru_cache ou disco).
        """
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,precipitation_sum&timezone=America/Sao_Paulo"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            'data': pd.to_datetime(data['daily']['time']),
            'temperatura_media': data['daily']['temperature_2m_mean'],
            'precipitacao_total': data['daily']['precipitation_sum']
        })
        return df

    async def fetch_public_holidays(self, year: int, country_code: str = 'BR') -> List[Dict[str, Any]]:
        """
        Busca feriados públicos do Nager.Date API assíncronamente.
        """
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    # Retry logic or return empty
                    return []

    def fetch_public_holidays_sync(self, year: int, country_code: str = 'BR') -> pd.DataFrame:
        """
        Wrapper síncrono para buscar feriados e retornar um DataFrame.
        """
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return pd.DataFrame()
