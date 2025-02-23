Análise de dados e segmentação de clientes

Esse código é uma aplicação completa para análise de dados e segmentação de clientes usando técnicas de machine learning, especificamente o algoritmo K-Means. Vou detalhar cada parte do código para que você possa entender melhor o que está sendo feito.

Importações
python
Copy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
pandas: Para manipulação de dados.

numpy: Para operações numéricas.

os: Para interagir com o sistema de arquivos.

matplotlib.pyplot e seaborn: Para visualização de dados.

KMeans: Algoritmo de clustering.

StandardScaler: Para normalização dos dados.

datetime: Para manipulação de datas.

Definição de Caminhos e Criação de Diretório
python
Copy
DATA_PATH = "/home/luiz/Downloads/Projetos_One/machine_learning/dataset_clientes_final.csv"
OUTPUT_DIR = "./analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH: Caminho do arquivo CSV contendo os dados.

OUTPUT_DIR: Diretório onde os resultados da análise serão salvos.

os.makedirs: Cria o diretório de saída se ele não existir.

Função load_data
python
Copy
def load_data():
    """Carrega o dataset e verifica se não está vazio."""
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8', sep=',')
        if df.empty:
            print(" O arquivo foi carregado, mas está vazio!")
            return None
        print(f" Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except Exception as e:
        print(f" Erro ao carregar o arquivo: {e}")
        return None
Carrega o dataset a partir do caminho especificado.

Verifica se o dataset está vazio.

Retorna o dataset ou None em caso de erro.

Função optimize_memory
python
Copy
def optimize_memory(df):
    """Reduz uso de memória convertendo tipos de dados."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
Converte colunas float64 para float32 e int64 para tipos menores (int32, int16, etc.).

Reduz o uso de memória do DataFrame.

Função clean_data
python
Copy
def clean_data(df):
    """Realiza limpeza de dados e otimização de memória."""
    initial_mem = df.memory_usage().sum() / 1024 ** 2
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')
    df = df.dropna(thresh=0.7 * len(df), axis=1)
    final_mem = df.memory_usage().sum() / 1024 ** 2
    print(f" Memória reduzida de {initial_mem:.2f} MB para {final_mem:.2f} MB")
    return df if df.shape[0] > 0 else None
Remove duplicatas.

Remove colunas que estão completamente vazias ou que têm mais de 30% de valores ausentes.

Calcula e exibe a redução de memória após a limpeza.

Função segment_clients
python
Copy
def segment_clients(df, n_clusters=5):
    """
    Segmenta clientes usando K-Means considerando todas as colunas numéricas.
    Adiciona a coluna 'Cluster' com os rótulos dos clusters.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print(" ERRO: Nenhuma coluna numérica encontrada para segmentação.")
        return df
    print(f" Usando {len(numeric_cols)} colunas para segmentação: {numeric_cols}")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    print(f" Segmentação concluída! Clientes classificados em {n_clusters} grupos.")
    return df
Seleciona colunas numéricas para segmentação.

Preenche valores ausentes com a média.

Normaliza os dados usando StandardScaler.

Aplica o algoritmo K-Means para segmentar os clientes em n_clusters grupos.

Adiciona uma coluna Cluster ao DataFrame com os rótulos dos clusters.

Função assign_cluster_names
python
Copy
def assign_cluster_names(df, col_name='categoria'):
    """
    Para cada cluster, extrai o valor mais frequente da coluna 'col_name'
    e retorna um dicionário mapeando o número do cluster para o nome real.
    Se a coluna não existir, usa rótulos padrão.
    """
    cluster_names = {}
    if col_name not in df.columns:
        print(f"Coluna '{col_name}' não encontrada. Usando labels padrão.")
        for cluster in df['Cluster'].unique():
            cluster_names[cluster] = f"Cluster {cluster}"
        return cluster_names

    for cluster in df['Cluster'].unique():
        mode_value = df[df['Cluster'] == cluster][col_name].mode()
        if not mode_value.empty:
            cluster_names[cluster] = mode_value.iloc[0]
        else:
            cluster_names[cluster] = f"Cluster {cluster}"
    return cluster_names
Atribui nomes aos clusters com base no valor mais frequente de uma coluna categórica (col_name).

Se a coluna não existir, usa rótulos padrão como "Cluster 0", "Cluster 1", etc.

Função comprehensive_eda
python
Copy
def comprehensive_eda(df):
    """
    Gera um EDA completo, explorando todas as colunas (numéricas e categóricas),
    salvando gráficos e estatísticas para fornecer um relatório rico.
    """
    # 1) Sumário de valores ausentes
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    missing_summary.to_csv(f"{OUTPUT_DIR}/missing_summary.csv")
    print(" Sumário de valores ausentes salvo em missing_summary.csv")

    # 2) Estatísticas descritivas para colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc_stats = df[numeric_cols].describe().T
        desc_stats.to_csv(f"{OUTPUT_DIR}/numeric_descriptive_stats.csv")
        print(" Estatísticas descritivas das colunas numéricas salvas em numeric_descriptive_stats.csv")

    # 3) Frequências para colunas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        freq_report = {}
        for c in cat_cols:
            freq = df[c].value_counts(dropna=False).head(50)  # Top 50 valores
            freq_report[c] = freq
            freq.to_csv(f"{OUTPUT_DIR}/freq_{c}.csv")
        print(" Frequências das colunas categóricas salvas em arquivos separados.")

    # 4) Matriz de correlação para colunas numéricas
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matriz de Correlação (todas as colunas numéricas)')
        plt.savefig(f"{OUTPUT_DIR}/correlation_matrix_all_numeric.png")
        plt.close()
        print(" Matriz de correlação de todas as colunas numéricas salva.")

    # 5) Gráficos de cada coluna
    # 5a) Para colunas numéricas: histograma e boxplot
    for col in numeric_cols:
        # Histograma
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.savefig(f"{OUTPUT_DIR}/hist_{col}.png")
        plt.close()

        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.savefig(f"{OUTPUT_DIR}/box_{col}.png")
        plt.close()

    # 5b) Para colunas categóricas: gráfico de barras com porcentagens (MODIFICADO)
    for col in cat_cols:
        plt.figure(figsize=(10, 6))
        # Calculando valores e porcentagens
        value_counts = df[col].value_counts(dropna=False).nlargest(20)
        total = len(df[col])
        percentages = (value_counts / total * 100).round(1)

        # Plotando
        ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")

        # Adicionando porcentagens
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3,
                    f'{percentages.iloc[i]}%',
                    ha="center", va="bottom", fontsize=9, rotation=45)

        plt.xticks(rotation=45, ha='right')
        plt.title(f'Distribuição de {col} (Top 20)')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/bar_{col}.png")
        plt.close()

    # 6) Sumários por cluster, se existir (MODIFICADO)
    if 'ClusterLabel' in df.columns:
        # Estatísticas numéricas por ClusterLabel
        cluster_numeric_mean = df.groupby('ClusterLabel')[numeric_cols].mean()
        cluster_numeric_mean.to_csv(f"{OUTPUT_DIR}/cluster_numeric_mean.csv")

        # Tamanho de cada cluster
        cluster_counts = df['ClusterLabel'].value_counts()
        cluster_counts.to_csv(f"{OUTPUT_DIR}/cluster_size.csv")

        # Gráfico de distribuição de clusters com porcentagens (NOVO)
        plt.figure(figsize=(10, 6))
        percentages = (cluster_counts / len(df) * 100).round(1)

        ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")

        # Adicionando porcentagens
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3,
                    f'{percentages.iloc[i]}%',
                    ha="center", va="bottom", fontsize=10)

        plt.title('Distribuição de Clientes por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Quantidade')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cluster_distribution.png")
        plt.close()

        # Boxplot de cada coluna numérica, separado por ClusterLabel
        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='ClusterLabel', y=col, data=df, palette='viridis')
            plt.title(f'Distribuição de {col} por Cluster')
            plt.savefig(f"{OUTPUT_DIR}/box_cluster_{col}.png")
            plt.close()

        print(" Sumários por cluster salvos (cluster_numeric_mean.csv, cluster_size.csv).")

    elif 'Cluster' in df.columns:
        # Mesma lógica, mas usando a coluna 'Cluster' se preferir
        cluster_numeric_mean = df.groupby('Cluster')[numeric_cols].mean()
        cluster_numeric_mean.to_csv(f"{OUTPUT_DIR}/cluster_numeric_mean.csv")

        cluster_counts = df['Cluster'].value_counts()
        cluster_counts.to_csv(f"{OUTPUT_DIR}/cluster_size.csv")

        # Gráfico de distribuição de clusters com porcentagens (NOVO)
        plt.figure(figsize=(10, 6))
        percentages = (cluster_counts / len(df) * 100).round(1)

        ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")

        # Adicionando porcentagens
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3,
                    f'{percentages.iloc[i]}%',
                    ha="center", va="bottom", fontsize=10)

        plt.title('Distribuição de Clientes por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Quantidade')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cluster_distribution.png")
        plt.close()

        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Cluster', y=col, data=df, palette='viridis')
            plt.title(f'Distribuição de {col} por Cluster')
            plt.savefig(f"{OUTPUT_DIR}/box_cluster_{col}.png")
            plt.close()

        print(" Sumários por cluster salvos (cluster_numeric_mean.csv, cluster_size.csv).")
Realiza uma análise exploratória completa dos dados.

Gera sumários de valores ausentes, estatísticas descritivas, frequências de valores categóricos, matriz de correlação, histogramas, boxplots, e gráficos de distribuição por cluster.

Salva os resultados em arquivos CSV e imagens PNG.

Função generate_report
python
Copy
def generate_report(df):
    """Gera um relatório textual final com as principais informações."""
    num_rows, num_cols = df.shape
    num_numeric = len(df.select_dtypes(include=[np.number]).columns)
    num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)

    # Se houver a coluna ClusterLabel ou Cluster, calculamos a distribuição
    cluster_col = 'ClusterLabel' if 'ClusterLabel' in df.columns else ('Cluster' if 'Cluster' in df.columns else None)
    if cluster_col:
        cluster_counts = df[cluster_col].value_counts(normalize=True) * 100
    else:
        cluster_counts = {}

    report_content = f"""
     Relatório de Análise - {datetime.now().strftime('%Y-%m-%d %H:%M')}
    ------------------------------------------------------------
    • Dimensões do dataset: {num_rows} linhas x {num_cols} colunas
    • Memória utilizada: {df.memory_usage().sum() / 1024 ** 2:.2f} MB
    • Colunas numéricas: {num_numeric}
    • Colunas categóricas: {num_categorical}

    • Distribuição dos Clusters:
    {cluster_counts.to_string() if cluster_col else 'Nenhuma coluna de cluster encontrada.'}
    """

    with open(f"{OUTPUT_DIR}/analysis_report.txt", "w") as f:
        f.write(report_content)

    print(f"\n Relatório salvo em: {OUTPUT_DIR}/analysis_report.txt")
Gera um relatório textual com as principais informações sobre o dataset, incluindo dimensões, uso de memória, número de colunas numéricas e categóricas, e distribuição dos clusters.

Salva o relatório em um arquivo de texto.

Função main
python
Copy
def main():
    df = load_data()
    if df is not None:
        df = optimize_memory(df)
        df = clean_data(df)
        if df is None:
            print(" Erro: O dataset ficou vazio após a limpeza. Revise os filtros.")
        else:
            # EXEMPLO: Segmentar e atribuir nomes
            df = segment_clients(df, n_clusters=4)  # Ajuste n_clusters conforme necessário
            if 'Cluster' not in df.columns:
                print(" Erro: A segmentação falhou. Verifique os dados de entrada.")
            else:
                # Se quiser nomes amigáveis baseados em alguma coluna, por exemplo 'categoria'
                # Se não tiver essa coluna, pode ignorar
                cluster_names = assign_cluster_names(df, col_name='categoria')
                df['ClusterLabel'] = df['Cluster'].map(cluster_names)

            # Agora fazemos a EDA completa
            comprehensive_eda(df)

            # E por fim, geramos o relatório textual
            generate_report(df)

            print(f"\n Análise concluída! Resultados salvos em: {OUTPUT_DIR}")
    else:
        print(" Erro: Não foi possível carregar os dados.")
Função principal que orquestra o carregamento, limpeza, segmentação, análise exploratória e geração de relatório.

Chama as funções na ordem correta e trata possíveis erros.

Execução do Script
python
Copy
if __name__ == "__main__":
    main()
