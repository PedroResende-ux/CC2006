import numpy as np

# Cria o board que será usado para o jogo, se usa numpy pela sua eficiencia computaciona
# 0 virá a simbolizar o espaço vazio
# X virá a simbolizar um espaço ocupado pelo vermelho
# Y virá a simbolizar um espaço ocupado pelo azul

class PopOutState:

    # Cria o board que será usado para o jogo, se usa numpy pela sua eficiencia computaciona
    # 0 simboliza o espaço vazio
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 'X' 

    #Checa se a coluna escolhida pode ser preenchida
    def is_valid_drop(self, column):
        if board[5][column]

    def apply_drop(self, column):
    
    def is_valid_pop(self, column):
        
        
    def apply_pop(self, column):
        

