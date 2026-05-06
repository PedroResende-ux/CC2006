import matplotlib.pyplot as plt
from ID3 import id3, test_accuracy, load_and_discretize_iris

def generate_learning_curve(data_path='iris.csv'):
    # Carrega e discretiza os dados
    df = load_and_discretize_iris(data_path)
    
    # Mistura os dados
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separa 20% para teste
    test_size = int(0.2 * len(df_shuffled))
    test_data = df_shuffled.iloc[:test_size].reset_index(drop=True)
    train_pool = df_shuffled.iloc[test_size:].reset_index(drop=True)
    
    training_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    accuracies = []
    
    features = list(train_pool.columns[:-1])
    target_col = "class"
    
    print("Gerando dados da Curva de Aprendizagem...")
    for size in training_sizes:
        # Pega só uma parte do treino
        subset_size = int(size * len(train_pool))
        train_subset = train_pool.iloc[:subset_size]
        
        # Treina e mede no teste
        tree = id3(train_subset, train_subset, features, target_col)
        
        acc = test_accuracy(test_data, tree)
        accuracies.append(acc)
        print(f"Tamanho do Treino: {size*100:.0f}% ({subset_size} amostras) -> Acurácia: {acc:.2f}%")
        
    # Plota o resultado
    plt.figure(figsize=(8, 5))
    plt.plot([s * 100 for s in training_sizes], accuracies, marker='o', linestyle='-', color='b')
    plt.title('Curva de Aprendizagem ID3 (Método Holdout)')
    plt.xlabel('Tamanho do Conjunto de Treino (%)')
    plt.ylabel('Proporção Correta no Conjunto de Teste (%)')
    plt.grid(True)
    plt.show()
