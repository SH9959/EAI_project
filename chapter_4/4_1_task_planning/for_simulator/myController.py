from ai2thor.controller import Controller
from typing import *
import numpy as np
from pprint import pprint

class DoneException(Exception):
    # print("task is over Done")
    pass

def parse_action(action_object:str):
    if(action_object == "DONE"):
        raise DoneException()
    else:
        action_object_pair = action_object.split("-")
        if len(action_object_pair) > 1:
            next_action = action_object_pair[0]  # Action name
            object_id = action_object_pair[1]    # Object name
    return next_action,object_id

class MyController(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
    def getImg(self):
        return self.last_event.cv2img

    # def dictStep(self,action:Dict):
        
    #     print(action)
    #     api_name = action["api_name"]
    #     if(api_name == "DONE"):
    #         raise DoneException()
    #     api_args = {args["name"]:args["value"] for args in action["api_args"] if args["name"]}
    #     stepargs = self.genAI2ThorAPI(api_name,api_args)
    #     env = super().step(**stepargs)
    #     print(env.metadata["lastActionSuccess"])
    #     if 'false' in str(env.metadata["lastActionSuccess"]):
    #         pprint("reason of failure: ",env.metadata['errorMessage'])
    #     pprint(stepargs)

    #     return env.cv2img
    
    def get_id_by_object_str(self, objtype:str):

        found = []
        # agent_pos = self.last_event.metadata["position"]
        # agent_pos = np.array(agent_pos["x"],agent_pos["y"],agent_pos["z"])

        for obj in self.last_event.metadata["objects"]:
            if(obj["objectType"].lower() == objtype.lower()):
                found.append(obj)

        return found[0]['objectId']
    
    def execute(self, action_object:str):
        next_action, object = parse_action(action_object)
        objectid = self.get_id_by_object_str(object)

        step_dict = self.genAI2ThorAPI(self,next_action,objectid)

        event = super().step(**step_dict)
        print(event.metadata["lastActionSuccess"])
        if 'false' in str(event.metadata["lastActionSuccess"]):
            pprint("reason of failure: ",event.metadata['errorMessage'])

        pprint(step_dict)

        return event.cv2img, event
    def genAI2ThorAPI(self,api_name:str,obj:str):
        api = getattr(self,api_name)
        return api(api_name, obj)

    def findObject(self,objtype:str):
        found = []

        for obj in self.last_event.metadata["objects"]:
            if(obj["objectType"].upper() == objtype.upper()):
                found.append(obj)

        if(len(found) == 0):
            print([obj["objectType"] for obj in self.last_event.metadata["objects"]])
            raise Exception(f"Object {objtype} Not Found")
        found = sorted(found,key = lambda o:o["distance"])
        return found

    def getTeleportPose(self,obj):
        '''Find the final state of the agent when it's willing to teleport'''        
        # ObjectType = obj["objectType"]
        # Pos = obj['position']
        # Rot = obj['rotation']

        # BBAxisAligned = obj['axisAlignedBoundingBox']
        # BBObjOriented = obj['objectOrientedBoundingBox']

        #reachable calculated by BFS and returned by simulator 
        # positions = Controller.step(action="GetReachablePositions").metadata["actionReturn"]
        poses = self.step(
            action="GetInteractablePoses",
            objectId=obj["objectId"],
            # positions=[dict(x=0, y=0.9, z=0)],
            # rotations=range(0, 360, 10),
            horizons=np.linspace(-30, 60, 30),
            standings=[True]
        ).metadata["actionReturn"]

        objpos = np.array([obj['position']["x"],obj['position']["y"],obj['position']["z"]])

        def distance(pos):
            pos = np.array([pos["x"],pos["y"],pos["z"]])
            return np.linalg.norm(objpos-pos)
            
        poses = sorted(poses,key=distance)[0]
        return {"x":poses["x"],"y":poses["y"],"z":poses["z"]},poses["rotation"],poses["horizon"],

    def PickupObject(self,objectId:str,forceAction:bool=True,placeStationary:bool=True):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"PickupObject",
            "objectId":obj["objectId"],
            "forceAction":forceAction,
            "manualInteract":placeStationary
        }
        return template

    def PutObject(self,objectId:str,forceAction:bool=True,placeStationary:bool=True):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"PutObject",
            "objectId":obj["objectId"],
            "forceAction":forceAction,
            "placeStationary":placeStationary
        }
        return template

    def OpenObject(self,objectId:str,openness:float=1):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"OpenObject",
            "objectId":obj["objectId"],
            "forceAction":True,
            "openness":openness
        }
        return template

    def CloseObject(self,objectId:str,forceAction:bool=True):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"CloseObject",
            "objectId":obj["objectId"],
            "forceAction":forceAction,
        }
        return template

    def SliceObject(self,objectId:str,forceAction:bool=True):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"SliceObject",
            "objectId":obj["objectId"],
            "forceAction":forceAction,
        }
        return template

    def ToggleObjectOn(self,objectId:str,forceAction:bool=True):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"ToggleObjectOn",
            "objectId":obj["objectId"],
            "forceAction":forceAction,
        }
        return template

    def ToggleObjectOff(self,objectId:str,forceAction:bool=True):
        obj = self.findObject(objectId)[0]
        template = {
            "action":"ToggleObjectOff",
            "objectId":obj["objectId"],
            "forceAction":forceAction,
        }
        return template

    def Teleport(self,target_place:str):
        obj = self.findObject(target_place)[0]
        position,rotation,horizon = self.getTeleportPose(obj)
        template = {
            "action":"Teleport",
            "position":position,
            "rotation":rotation,
            "horizon":horizon,
            "standing":True
        }
        return template

def get_all_objects_in_scene(event):
    l = len(event.metadata['objects'])
    all_objects = list(set([event.metadata['objects'][i]['name'].split("_")[0] for i in range(l)]))
    return all_objects

def get_prompt(
        instruction:str, 
        all_objects:List[str], 
        skill_set:Dict,
        **kwargs
        ) -> str:
    
    prompt = f"""
    You are a household robot that can interact with objects in a 3D environment.
    You are given a task to complete:
    ###{instruction}
    You are given a list of objects in the scene:
    ###{str(all_objects)}
    Your task is to complete the task step by step and you can only use the following APIs to finish your task:
    ###{str(skill_set)}
    Please output all the steps you need to take to complete the task in the python list format by order. for example if you want to pick up something you should output like "PickupObject-Potato".

    """

    if 'history_of_actions' in kwargs:
        prompt += f"""
        You have already done the following steps:
        ###{str(kwargs['history_of_actions'])}
        """

    if 'current_image' in kwargs:
        prompt += f"""
        Now you can see the image in front of you:
        """

    return prompt





if __name__ == "__main__":
    p  = MyController()

    import json
    with open("low.json","rt") as f:
        j = json.load(f)
    act_idx = 0
    act_api = j[0]["OUTPUT"]["api_list"]
    def getAction(img:Union[str, np.ndarray],text:str) -> Dict:
        global act_idx
        act_idx += 1
        return act_api[act_idx-1]
    
    p.reset("FloorPlan418")

    done = False
    img = p.getImg()
    import time
    while not done:
        time.sleep(1)
        act = getAction(img,"Test")
        img = p.dictStep(act)
    
    input()
