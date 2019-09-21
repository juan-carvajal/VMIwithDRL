from agent_model.model import Model
from agent_model.training_agent import TrainingAgent


class VMI(Model):
#LOS ESTADOS DE ESTA CLASE SON LOS NIVELES DE INVENTARIOS QUE TIENE PARA CADA UNA DE LAS CADUCIDADES
#LAS ACCIONES POSIBLES VAN DESDE 0 HASTA MAX_A
    def __init__(self,hospitals,max_A , shelf_life,initial_state=None):
        super(VMI, self).__init__(initial_state, max_A, shelf_life)
        self.day=1



    #def model_logic(self, state, action):
