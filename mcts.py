import math
import random
import copy


# mcts.py

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        # Guardar estado, pai e movimento que gerou este nó.
        self.state = state
        self.parent = parent
        self.move = move  # ("drop", col) ou ("pop", col)

        # Estatísticas do nó
        self.children = []
        self.visits = 0
        self.value = 0.0

        # Lista de movimentos não tentados (preencher ao criar o nó)
        self.untried_moves = []


class MCTS:
    """MCTS com UCT e opções de expansão configuráveis.

        Parâmetros:
            - iterations: número de simulações por chamada a search.
            - exploration_weight: constante C usada na fórmula UCT.
            - num_children_to_expand: número de filhos a expandir por nó.

    Requisitos de implementação (ações concretas):
        1) Seleção: implementar UCT exatamente como especificado:
            UCT(child) = (child.value / child.visits) + C * sqrt( ln(parent.visits) / child.visits )
            - Tratar child.visits == 0 como +inf para forçar exploração inicial.

    2) Expansão: criar até k = num_children_to_expand filhos
         para um nó quando existirem movimentos não tentados.
                 - Para cada novo filho: copiar estado (deepcopy), aplicar
                     ação ("drop" ou "pop"), inicializar estatísticas e anexar.

      3) Simulação: do estado do nó expandido, executar rollout
         até terminal. Em cada passo do rollout, escolher aleatoriamente
         entre movimentos válidos (drops e pops).
         - Respeitar regras do PopOut em todas as transições.

      4) Retropropagação: subir até à raiz atualizando `visits`
         e `value` de cada nó. Definir `value` consistentemente
         (por exemplo, soma de pontos onde +1 = vitória para o
         jogador da raiz, 0 = derrota, 0.5 = empate) e aplicar.

        5) Exportação de dataset: registar pares (estado_flat, movimento)
            durante as simulações e fornecer export_dataset(filename)
            que escreva CSV com 42 colunas + class.

        6) Interface: search(state) deve devolver um movimento no formato
            ("drop", col) ou ("pop", col). search deve suportar ser
            invocado em modos human vs cpu e cpu vs cpu.
    """

    def __init__(self, iterations=1000, exploration_weight=1.41, num_children_to_expand=1):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.num_children_to_expand = num_children_to_expand
        self._dataset_rows = []  # armazenar pares (estado_flat, movimento)

    def search(self, initial_state):
        """Executar MCTS e devolver melhor movimento.

        Implementar o ciclo: seleção -> expansão -> simulação -> retroprop.
        No fim, retornar o movimento do filho com maior visits.
        """

    def _select(self, node):
        """Descer a árvore usando UCT até nó para expandir ou terminal."""

    def _expand(self, node, k=1):
        """Criar até k filhos a partir de movimentos não tentados."""

    def _simulate(self, state):
        """Executar rollout aleatório (respeitar drop/pop) até terminal."""

    def _backpropagate(self, node, result):
        """Atualizar `visits` e `value` desde `node` até à raiz."""

    def export_dataset(self, filename="popout_dataset.csv"):
        """Gravar `self._dataset_rows` em CSV: 42 colunas + class."""
