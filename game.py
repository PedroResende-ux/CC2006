import numpy as np

# Cria o board que será usado para o jogo, se usa numpy pela sua eficiencia computaciona
# 0 virá a simbolizar o espaço vazio
# 1 virá a simbolizar um espaço ocupado pelo jogador 1
# 2 virá a simbolizar um espaço ocupado pelo jogador 2

class PopOutState:

    # Cria o board que será usado para o jogo, se usa numpy pela sua eficiencia computaciona
    # 0 simboliza o espaço vazio
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = '1' 

    #Checa se a coluna escolhida pode ser preenchida
    def is_valid_drop(self, column):
        return self.board[0][column] == 0

    #adiciona a peça do jogador atual na coluna
    def apply_drop(self, column):
        if self.is_valid_drop(column):
            for i in range(5, -1, -1):
                if self.board[i][column] == 0:
                    self.board[i][column] = int(self.current_player)
                    self.current_player = '2' if self.current_player == '1' else '1'
                    return True
        return False

    
    #Checa se é possivel aplicar pop (remover a peça do jogador na base da coluna)
    def is_valid_pop(self, column):
        return self.board[5][column] != 0
        
    #Aplica o pop
    def apply_pop(self, column):
        if self.is_valid_pop(column):
            if self.board[5][column] == int(self.current_player):
                self.board[5][column] = 0
                self.current_player = '2' if self.current_player == '1' else '1'
                return True
        return False

    #checa se houve vitoria
    def check_win(self, player):
        player_id = int(player)
        
        # Checa a horizontal
        for row in range(6):
            for col in range(4):
                if all(self.board[row][col + i] == player_id for i in range(4)):
                    return True
        
        # Checa a vertical
        for col in range(7):
            for row in range(3):
                if all(self.board[row + i][col] == player_id for i in range(4)):
                    return True
        
        # Checa a primeira diagonal
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i][col + i] == player_id for i in range(4)):
                    return True
        
        # Checa a segunda diagonal
        for row in range(3):
            for col in range(3, 7):
                if all(self.board[row + i][col - i] == player_id for i in range(4)):
                    return True
        
        return False


