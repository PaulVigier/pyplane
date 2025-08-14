from libs import constants as cst
from libs import atm
print(700*cst.FPS_TO_KT)
print(atm.TAStoCAS(700*cst.FPS_TO_KT, 20000, 0,Offset=True))