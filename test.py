from libs import constants as cst
from libs import atm
from libs import geo
from libs import das
from libs import tacview

data,_=das.das_read(r"data\Have dream\TM check\20250818_Jager_382_Dream Ck.csv",'T-38')
tacview.das2tacview("test",data)