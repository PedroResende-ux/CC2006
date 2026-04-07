"""Cenarios de teste rapido para o jogo PopOut.

Formato de peca:
- 0 = vazio
- 1 = jogador 1
- 2 = jogador 2

"""

# tabuleiro vazio
board_vazio = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]

# vitoria horizontal para o jogador 1
board_vitoria_j1 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
]

# tabuleiro cheio com pops possiveis para ambos os jogadores
# este cenario deve permitir acao empate opcional (alem de pop)
board_cheio_com_pop = [
    [1, 2, 1, 2, 1, 2, 1],
    [2, 1, 2, 1, 2, 1, 2],
    [1, 2, 1, 2, 1, 2, 1],
    [2, 1, 2, 1, 2, 1, 2],
    [1, 2, 1, 2, 1, 2, 1],
    [1, 2, 1, 2, 1, 2, 1],
]

# tabuleiro cheio sem pop legal para o jogador 1 (base so com pecas do jogador 2)
# este cenario deve resultar em empate forcado para o jogador 1
board_cheio_sem_pop_j1 = [
    [1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2],
]

# tripla repeticao: lista de estados que se repetem que permitem o empate
estado_repeticao_a = (
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0],
    ],
    "1",
)

# A sequencia que permite draw consiste em o 
# mesmo jogador no mesmo tabuleiro 3 vzs ou o mesmo tabuleiro em 3 vzs seguidas?
sequencia_repeticao_tripla = [
    estado_repeticao_a,
    estado_repeticao_a,
    estado_repeticao_a,
]