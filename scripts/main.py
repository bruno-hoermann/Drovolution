import bge
import mathutils
import numpy as np
import nn
from time import monotonic

cont = bge.logic.getCurrentController()
own = cont.owner
scene = bge.logic.getCurrentScene()
obj1 = scene.objectsInactive['Cube']
obj2 = scene.objectsInactive['Zielmarkierung']
spawn=scene.objects['spawn']
step = scene.objects['step']



layer_dims=[6,4,3,2]

own["tempTime"]=0
own["Gen"]=1
own["bestscore"]=32000000
own["numDrones"]=50

bge.logic.drone = [0] * own["numDrones"]
d=bge.logic.drone
spawn.localPosition=mathutils.Vector((0, 0, 0))
print("Main")
ziel=mathutils.Vector((0, 2, 5))
zielm=scene.addObject(obj2,spawn,0)
zielm.localPosition+=ziel
for i in range(own["numDrones"]):
   
    d[i]=scene.addObject(obj1,spawn,0)
    d[i]["ID"]=i
    d[i]["istPos"]=spawn.worldPosition-mathutils.Vector((0, 0, 0))
    d[i]["orientierung"]=spawn.localOrientation
    d[i]["sollPos"] = spawn.worldPosition+ziel
    d[i]["parameters"]=nn.initialize_parameters_deep(layer_dims)
    spawn.localPosition+=mathutils.Vector((-10, 0, 0.0))
    


def reset():
    for i in range(own["numDrones"]):
        d[i].localPosition=d[i]["istPos"]
        d[i].localLinearVelocity=[0,0,0]
        d[i].localAngularVelocity=[0,0,0]
        d[i].localOrientation=d[i]["orientierung"]
        d[i]["score"]=0
        
    
def run():
    if step.localPosition.y==0:
        reset()
        step.localPosition.y=1
      

    if step.localPosition.y ==1:
       
        if bge.logic.getFrameTime()<own["tempTime"] + 10 :
            testgen()

        else: 
            step.localPosition.y =2
            own["tempTime"]=bge.logic.getFrameTime()
    if step.localPosition.y ==2:
        breed()
        print(own["numDrones"])
        step.localPosition.y=0
    if step.localPosition.y==5:
        reset()
        step.localPosition.y=6
    if step.localPosition.y==6:
        testgen(True)

    
    
def breed():
    print("Generation:", own["Gen"])
    own["Gen"]+=1
    best_id=calcbest()
    
    #d[0]["parameters"]=d[best_id]["parameters"]
    
    for i in range(0,own["numDrones"]):
        d[i]["parameters"]=nn.vary_parameters_deep(layer_dims, d[best_id]["parameters"], 0.005 )
        d[i]["parameters"]=nn.vary_parameters_deep_ga_delete(layer_dims, d[i]["parameters"], -5)
        
    
    
        


def testgen(displayOut=False):
    for i in range(own["numDrones"]):
        #print(own.localOrientation.to_euler()[0])
        d[i]["X"]=np.array([[\
        d[i]["sollPos"][1]-d[i].localPosition[1],\
        d[i]["sollPos"][2]-d[i].localPosition[2],\
        d[i].localLinearVelocity[1],\
        d[i].localLinearVelocity[2],\
        d[i].localOrientation.to_euler()[0],\
        d[i].localAngularVelocity[0]]]).T

        out=nn.L_model_forward(d[i]["X"], d[i]["parameters"],"tanh")
           
        d[i].applyTorque([90*(0.5-out[0]),0,0],True)
        d[i].applyForce([0,0,30*out[1]],True)
        
        d[i]["score"]+= np.linalg.norm(d[i].localPosition-d[i]["sollPos"])
        #print("d",i,":", d[i]["score"],end='')
        if displayOut and i == 0: print(out.T)

def calcbest():
    lim=32000000
    for i in range(own["numDrones"]):
        if d[i]["score"]<lim:
            best=d[i]
            lim=d[i]["score"]
        if d[i]["score"]>32000000:
            print("riesiger score")
            
    print("best drone is: ", best["ID"], "\nwith a score of:", round(best["score"],1))
    
    if best["score"]<own["bestscore"]:
        own["bestpara"]=best["parameters"]
        own["bestscore"]=best["score"]
    
    return (best["ID"])


def calcbest2():
    
    lim=32000000
    best=[0]
    
    for i in range(own["numDrones"]):
        if d[i]["score"]<lim:
            ndbest=best
            best=d[i]
            lim=d[i]["score"]
        if d[i]["score"]>32000000:
            print("riesiger score")
            
    print("lowest score:", round(best["score"],1))
    
    if best["score"]<own["bestscore"]:
        own["bestpara"]=best["parameters"]
        own["bestscore"]=best["score"]
    
    return best["ID"],ndbest["ID"]


def switchbest():
    own["numDrones"]=1
    d[0]["parameters"]=own["bestpara"]
    d[0]["score"]=own["bestscore"]
    print("Winner mit:", d[0]["score"])
    step.localPosition.y=5

def breed2():
    print("Generation:", own["Gen"])
    own["Gen"]+=1
    best_id,ndbest_id=calcbest2()
    
    d[0]["parameters"]=d[best_id]["parameters"] #nur für visualisierung
    
    for i in range(1,own["numDrones"]):
        d[i]["parameters"]=nn.vary_parameters_deep_ga_cross(layer_dims, d[best_id]["parameters"], d[ndbest_id]["parameters"], 0)
        d[i]["parameters"]=nn.vary_parameters_deep(layer_dims, d[best_id]["parameters"], 0.005 )
        #d[i]["parameters"]=nn.vary_parameters_deep_ga_delete(layer_dims, d[i]["parameters"], -1.8)
            
    
def breed3():
    
    for i in range(own["numDrones"]):
        scores[i]=d[i]["score"]
        
    sorted=np.argsort(scores)
    print(sorted)
    del sorted[25:own["numDrones"]]