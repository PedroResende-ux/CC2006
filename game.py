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
        self.last_move_type = None
        self.last_move_row = None
        self.last_move_col = None
        self.last_mover = None
        self.is_draw = False
        self.draw_reason = None
        self.position_counts = {}

    def _state_key(self):
        # isto aqui vai servir depois para ver se o mesmo estado ja apareceu varias vezes
        # a ideia e guardar o tabuleiro + o jogador que vai jogar
        # nao tenho a certeza ainda se isto vai ficar assim mesmo, mas ja deixa o caminho feito
        board_key = self.board.tobytes()
        player_key = self.current_player

        
        return (board_key, player_key)

    #Checa se a coluna escolhida pode ser preenchida
    def is_valid_drop(self, column):
        return self.board[0][column] == 0

    #adiciona a peça do jogador atual na coluna
    def apply_drop(self, column):
        if self.is_valid_drop(column):
            mover = self.current_player
            for i in range(5, -1, -1):
                if self.board[i][column] == 0:
                    self.board[i][column] = int(mover)
                    self.last_move_type = 'drop'
                    self.last_move_row = i
                    self.last_move_col = column
                    self.last_mover = mover
                    self.current_player = '2' if self.current_player == '1' else '1'
                    return True
        return False

    
    #Checa se é possivel aplicar pop (remover a peça do jogador na base da coluna)
    def is_valid_pop(self, column):
        return self.board[5][column] == int(self.current_player)
        
    #Aplica o pop se a peça na base da coluna for do jogador atual
    def apply_pop(self, column):
        if self.is_valid_pop(column):
            mover = self.current_player
            if self.board[5][column] == int(mover):
                for row in range(5, 0, -1):
                    self.board[row][column] = self.board[row - 1][column]
                self.board[0][column] = 0
                self.last_move_type = 'pop'
                self.last_move_row = None
                self.last_move_col = column
                self.last_mover = mover
                self.current_player = '2' if self.current_player == '1' else '1'
                return True
        return False

    def _has_four(self, player_id):
        # horizontal
        for row in range(6):
            for col in range(4):
                if all(self.board[row][col + i] == player_id for i in range(4)):
                    return True

        # vertical
        for row in range(3):
            for col in range(7):
                if all(self.board[row + i][col] == player_id for i in range(4)):
                    return True

        # diagonal principal
        for row in range(3):
            for col in range(4):
                if all(self.board[row + i][col + i] == player_id for i in range(4)):
                    return True

        # diagonal secundaria
        for row in range(3):
            for col in range(3, 7):
                if all(self.board[row + i][col - i] == player_id for i in range(4)):
                    return True

        return False

    # checa vitoria no tabuleiro inteiro
    def check_win(self, player):
        player_id = int(player)
        other_id = 1 if player_id == 2 else 2

        player_has_four = self._has_four(player_id)
        other_has_four = self._has_four(other_id)

        # regra do PopOut: em caso de vitoria simultanea apos pop, vence quem jogou
        if self.last_move_type == 'pop' and player_has_four and other_has_four:
            return str(player) == self.last_mover

        return player_has_four


