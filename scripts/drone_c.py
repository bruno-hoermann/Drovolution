import bge
import mathutils
import numpy as np
import nn
from time import monotonic



cont=bge.logic.getCurrentController()
own=cont.owner
move = cont.actuators["move"]

#building neural net
parameters=nn.initialize_parameters_deep([3,6,2])

#setting Timer
if "init" not in own:
    own["init"] = True
    own["time"] = monotonic()

own["timePassed"] = monotonic() - own["time"]


thrust= 40
p
npPos=np.array([0,0,0])
npPos=own.localPosition
own["score"]=0.0
own["m"]=0

def fly():
    
  
    #print(own.localOrientation.to_euler()[0])
    X=np.array([[own.localPosition[2],own.localLinearVelocity[1],own.localLinearVelocity[2]]]).T
    #rforce=sollPos-own.localPosition
    out=nn.L_model_forward(X, parameters)
   
    move.torque = [30*(0.5-out[0]),0,0]
    move.force =[0,0,20*out[1]]
    cont.activate(move)
    own["timePassed"] = monotonic() - own["time"]
    print(own["timePassed"])
    own["score"]+= np.linalg.norm(own.localPosition-sollPos)
    own["m"]+=1
    print(own["score"]/own["m"])
        
