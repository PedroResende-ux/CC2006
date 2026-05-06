"""Interface simples de terminal para o PopOut.

Mostra o tabuleiro em ASCII e traduz:
- 0 -> espaco vazio
- 1 -> X
- 2 -> O

Nao usa tkinter nem outra GUI grafica.

TODO / O QUE FALTA :
- loop de jogo real nao existe, so mostra um tabuleiro de demo
- precisa de input do jogador (qual coluna fazer drop/pop)
- integrar com o MCTS pra CPU jogar automaticamente
- tratar quando jogador ganha, perde ou empata (3-fold repetition)
- melhorar a leitura do tabuleiro, colocar uns numeros pras colunas
- validacao de input (nao deixa numero invalido nem coluna cheia)
- menu de opcoes tipo "human vs cpu", "cpu vs cpu" etc
- talvez adicionar delay visual pra nao ficar muito rapido
- gui.py ta vazio entao fica pra depois :P
"""

import os

from game import PopOutState


class TerminalPopOutUI:
    def __init__(self):
        self.game = PopOutState()
        self.symbols = {
            0: " ",
            1: "X",
            2: "O",
        }

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def render_board(self):
        self.clear_screen()
        print("PopOut")
        print()
        print("   1   2   3   4   5   6   7")
        print(" +---+---+---+---+---+---+---+")

        for row in range(6):
            line = " |"
            for col in range(7):
                value = int(self.game.board[row][col])
                line += f" {self.symbols[value]} |"
            print(line)
            print(" +---+---+---+---+---+---+---+")

        print()
        print(f"Jogador atual: {self.symbols[int(self.game.current_player)]}")

    def demo_board(self):
        # tabuleiro semi-completo so para mostrar a interface
        self.game.board[2][2] = 1
        self.game.board[3][1] = 2
        self.game.board[3][2] = 1
        self.game.board[4][0] = 2
        self.game.board[4][1] = 1
        self.game.board[4][2] = 2
        self.game.board[5][0] = 1
        self.game.board[5][1] = 2
        self.game.board[5][2] = 1
        self.game.board[5][3] = 2

    def run(self):
        self.demo_board()
        self.render_board()
        print()
        print("Por enquanto serve para mostrar X e O no tabuleiro.")
        print()


if __name__ == "__main__":
    app = TerminalPopOutUI()
    app.run()
