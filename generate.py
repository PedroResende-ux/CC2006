# generate.py

# Este ficheiro era para gerar o dataset de treino com MCTS a jogar
# PopOut contra si mesmo. Aqui fica a versão comentada, mais direta,
# para lembrar o fluxo sem deixar a implementação escrita.

# O que este script precisa fazer:
# - criar jogos automáticos com PopOutState
# - pedir jogadas ao MCTS
# - guardar pares estado + movimento
# - exportar tudo para popout_dataset.csv

# Fluxo esperado:
# 1) começar com um estado vazio do tabuleiro
# 2) criar um agente MCTS com o número de iterações definido
# 3) enquanto o jogo não acabar:
#    - verificar vitória, derrota ou empate
#    - pedir a melhor jogada ao MCTS
#    - transformar o tabuleiro numa lista de 42 valores
#    - guardar a jogada escolhida no formato "drop_3" ou "pop_2"
#    - aplicar a jogada no estado atual
# 4) no fim, criar um DataFrame com colunas pos_0 até pos_41 + class
# 5) gravar o CSV em popout_dataset.csv


def generate_data(num_games=20, mcts_iterations=200):
    # Implementar a geração dos dados aqui.
    # num_games controla quantos jogos automáticos são simulados.
    # mcts_iterations controla quantas simulações o MCTS faz por jogada.
    raise NotImplementedError("generate.py ainda está só como guia comentado.")


if __name__ == "__main__":
    # Ligar isto depois à função real quando a geração estiver pronta.
    raise NotImplementedError