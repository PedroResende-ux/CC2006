# PopOut

Variante do Connect-4 com movimentos *pop*, e dois jogadores artificiais — um baseado em **Monte Carlo Tree Search** e outro numa **árvore de decisão ID3** treinada em self-play do MCTS.

Trabalho da unidade curricular de Inteligência Artificial, FCUP, 2025/2026.

**Autores:** Miguel Almeida, Francisco Passos, Pedro Resende.

---

## O que está aqui

- O jogo PopOut, com as três regras especiais do enunciado (vitória simultânea após pop, empate por tabuleiro cheio, repetição tripla).
- Um *game loop* que suporta os três modos pedidos: humano-vs-humano, humano-vs-computador, computador-vs-computador.
- Um jogador MCTS com UCT, propagação de subárvores resolvidas e *only-move shortcut*. O rollout aleatório é JIT-compilado em Numba.
- Uma árvore de decisão ID3 implementada de raiz (sem `scikit-learn`), treinada em dois datasets:
  - O *Iris* como warm-up (discretização por *threshold* óptimo de *information gain*).
  - Um dataset gerado por self-play do MCTS, em que cada linha é um par `(estado, melhor jogada segundo o MCTS)`.
- Um *script* para gerar o dataset de self-play em paralelo com múltiplos workers.
- Um notebook que coordena as experiências, os resultados e a análise.

---

## Estrutura

```
.
├── game/                       Motor do jogo + game loop + jogadores
│   ├── board.py                PopOutBoard
│   ├── game.py                 PopOutGame
│   ├── player.py               Player (ABC) + HumanPlayer + RandomPlayer
│   └── display.py              Renderização ASCII
├── ai/                         Algoritmos de IA
│   ├── mcts.py                 MCTS + UCT + propagação terminal
│   ├── mcts_player.py          MCTSPlayer(Player)
│   ├── rollout_numba.py        Rollout JIT-compilado
│   └── decision_tree.py        ID3
├── scripts/
│   └── generate_dataset.py     Gerador de dataset por self-play
├── tests/                      Suite pytest
├── data/
│   └── iris.csv
├── notebooks/
│   └── main.ipynb              Notebook principal
├── main.py                     Menu interactivo
├── requirements.txt
└── README.md
```

Datasets gerados em `data/` (com excepção do `iris.csv`) são excluídos do controlo de versões — regeneram-se a partir do *script* e do *seed*.

---

## Setup

Requer **Python ≥ 3.10**. A partir da raiz do repositório:

```bash
python -m pip install -r requirements.txt
```

---

## Como correr

### Testes

```bash
# Rápidos
python -m pytest tests/ -v -m "not slow"

# Suite completa (inclui benchmarks lentos)
python -m pytest tests/ -v
```

### Jogar uma partida

```bash
python main.py
```

Apresenta o menu com os três modos.

### Gerar o dataset PopOut

```bash
python -m scripts.generate_dataset \
    --games 200000 \
    --iterations 5000 \
    --workers 16 \
    --out data/popout_200k.csv \
    --base-seed 0
```

O *script* aceita também `--smoke`, que valida o pipeline em ~30 s sem produzir output final.

### Notebook

```bash
jupyter notebook notebooks/main.ipynb
```

O notebook importa os módulos `.py`, corre as experiências (treino do ID3, comparação MCTS vs ID3, variantes do MCTS) e apresenta os resultados.

---

## Notas de implementação

- **MCTS — UCT com `c = √2`.** A perspectiva do *exploitation* é invertida (`1 − mean`) nos nós em que joga o adversário. Nós cujo desfecho é provado (vitória, derrota ou empate forçado) são marcados e excluídos das iterações seguintes. Existe ainda um *only-move shortcut* para posições com uma única jogada legal.
- **Rollout.** Uniformemente aleatório, com limite de profundidade de 80 plies. A versão em Numba é validada por um teste de equivalência estatística contra a versão pura em Python.
- **ID3.** Entropia e *information gain* calculados em NumPy. Iris é tratado com discretização binária por *threshold* óptimo. Para o dataset PopOut, a árvore é treinada sobre as 42 células do tabuleiro mais um conjunto de features derivadas (`move_count`, jogador a mover, contagens da linha inferior).
- **Dataset.** Gerado por self-play do MCTS, com aberturas aleatórias de 0–12 plies por jogo e ε-greedy a 25% (jogada aleatória não rotulada / 75% jogada do MCTS rotulada). Só as jogadas do MCTS aparecem como linhas no CSV. Determinismo é garantido pelo `--base-seed`, independentemente do número de workers.

---

## Decisões e simplificações documentadas

- **Regra 2 (tabuleiro cheio: pop ou empate)** — resolvida no game loop por delegação ao `Player.choose_pop_or_draw`. O `MCTSPlayer` herda o default `"pop"` e nunca declara empate voluntariamente. A modelação da escolha binária dentro da árvore do MCTS ficou fora do escopo.
- **Regra 3 (repetição tripla)** — resolvida no game loop por delegação ao `Player.choose_continue_or_draw`. Dentro da árvore do MCTS, uma posição que apareça três vezes é tratada como `TERMINAL_DRAW` (conservador, mas seguro). O rollout em Numba não verifica repetição; o limite de profundidade de 80 plies impede *loops* infinitos.
