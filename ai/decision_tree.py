import numpy as np
import pandas as pd
import pprint

 # ==========================================
# 1. FUNÇÕES MATEMÁTICAS PARA ID3
# ==========================================

def entropy(target_col):
    """Calcula a entropia de um conjunto de dados."""
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy_val

def information_gain(data, split_attribute_name, target_name):
    """Calcula o ganho de informação ao dividir os dados em um determinado atributo."""
    # Entropia total do conjunto de dados
    total_entropy = entropy(data[target_name])
    
    # Calcula a entropia ponderada para a divisão
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    # Ganho de Informação
    information_gain_val = total_entropy - weighted_entropy
    return information_gain_val

# ==========================================
# 2. ALGORITMO ID3 PRINCIPAL
# ==========================================

def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    """
    Constrói a árvore de decisão recursivamente.
    """
    # Caso Base 1: Se todos os valores alvo forem iguais, retorna essa classificação
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # Caso Base 2: Se o conjunto de dados estiver vazio, retorna o valor alvo mais comum do conjunto de dados original
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # Caso Base 3: Se não houver mais atributos para dividir, retorna o valor alvo mais comum do conjunto de dados atual
    elif len(features) == 0:
        return parent_node_class
    
    # Caso Recursivo: Cresce a árvore
    else:
        # Determina a classe do nó pai (classe mais comum dos dados atuais)
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Seleciona o atributo que melhor divide o conjunto de dados (maior ganho de informação)
        item_values = [information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Cria a estrutura da árvore como um dicionário aninhado
        tree = {best_feature: {}}
        
        # Remove o melhor atributo do espaço de atributos
        features = [i for i in features if i != best_feature]
        
        # Cresce um ramo sob o nó raiz para cada valor possível do atributo raiz
        for value in np.unique(data[best_feature]):
            value = value
            # Divide o conjunto de dados
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # Chama o algoritmo ID3 recursivamente
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            
            # Adiciona a subárvore sob a raiz
            tree[best_feature][value] = subtree
            
        return tree

# ==========================================
# 3. PREDIÇÃO & AVALIAÇÃO
# ==========================================

def predict(query, tree, default='Iris-setosa'):
    """Prediz a classe para uma única consulta usando a árvore gerada."""
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
    return default

def test_accuracy(data, tree):
    """Calcula a acurácia da árvore em um conjunto de teste."""
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i], tree)
    
    accuracy = (np.sum(predicted["predicted"].values == data["class"].values) / len(data)) * 100
    return accuracy

# ==========================================
# 4. PRÉ-PROCESSAMENTO DE DADOS
# ==========================================

def optimal_binary_split(df, feature, target_col):
    """Encontra o limiar que maximiza o ganho de informação para um atributo contínuo."""
    best_ig = -1
    best_threshold = None
    
    # Ordena os valores únicos para encontrar possíveis pontos de divisão (meios)
    unique_vals = np.sort(df[feature].unique())
    
    for i in range(len(unique_vals) - 1):
        threshold = (unique_vals[i] + unique_vals[i+1]) / 2.0
        
        # Cria uma coluna binária temporária
        temp_col = df[feature] <= threshold
        temp_df = df.copy()
        temp_df['temp_split'] = temp_col
        
        # Calcula o ganho de informação para esta divisão binária específica
        ig = information_gain(temp_df, 'temp_split', target_col)
        
        if ig > best_ig:
            best_ig = ig
            best_threshold = threshold
            
    return best_threshold

def load_and_discretize_iris(filepath):
    """Loads CSV and applies optimal binary quantization using Information Gain."""
    df = pd.read_csv(filepath)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        
    features = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    
    # Calcula o limiar ótimo para cada atributo contínuo
    for feature in features:
        threshold = optimal_binary_split(df, feature, 'class')
        print(f"Optimal binary split for {feature} is at <= {threshold:.2f}")
        
        # Aplica a divisão binária: 'Low' e 'High'
        df[feature] = np.where(df[feature] <= threshold, 'Low', 'High')
        
    return df

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

if __name__ == '__main__':
    # 1. Carrega e discretiza os dados
    print("Carregando e discretizando o conjunto de dados Iris...")
    df = load_and_discretize_iris('iris.csv')
    
    # 2. Divide em conjuntos de treino (80%) e teste (20%)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(0.8 * len(df_shuffled))
    
    train_data = df_shuffled.iloc[:split_index]
    test_data = df_shuffled.iloc[split_index:]
    
    # 3. Treina a árvore
    print("Treinando a Árvore de Decisão ID3...")
    features = list(train_data.columns[:-1])
    target_col = "class"
    
    tree = id3(train_data, train_data, features, target_col)
    
    # 4. Exibe visualmente e avalia
    print("\n--- Árvore de Decisão Gerada ---")
    pprint.pprint(tree)
    
    accuracy = test_accuracy(test_data, tree)
    print(f"\nAcurácia no Conjunto de Teste: {accuracy:.2f}%")