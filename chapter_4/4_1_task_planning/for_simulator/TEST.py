from play import Player, DoneException
from typing import Dict
import json

class ActionExecutor:
    def __init__(self):
        self.player = Player()
        
    def parse_action(self, action_str: str) -> Dict:
        """解析动作字符串为ai2thor需要的字典格式"""
        if action_str == "DONE":
            return {"api_name": "DONE", "api_args": []}
            
        action_parts = action_str.split('-', 1)
        action_type = action_parts[0]
        arg = action_parts[1] if len(action_parts) > 1 else ""

        action_map = {
            "TeleportObject": ("Teleport", {"target_place": arg}),
            "PickupObject": ("PickupObject", {"objectId": arg}),
            "PutObject": ("PutObject", {
                "objectId": arg,
                "forceAction": True,
                "placeStationary": True
            }),
            "OpenObject": ("OpenObject", {
                "objectId": arg,
                "openness": 1.0
            }),
            "CloseObject": ("CloseObject", {"objectId": arg}),
            "ToggleObjectOn": ("ToggleObjectOn", {"objectId": arg}),
            "ToggleObjectOff": ("ToggleObjectOff", {"objectId": arg}),
            "SliceObject": ("SliceObject", {"objectId": arg}),
        }

        api_name, api_args = action_map[action_type]
        return {
            "api_name": api_name,
            "api_args": [{"name": k, "value": v} for k, v in api_args.items()]
        }

    def execute(self, scene: str, plan: Dict):
        """执行完整的动作序列"""
        self.player.reset(scene)
        
        try:
            for step_key in sorted(plan.keys(), key=lambda x: int(x)):
                actions = plan[step_key].split(',')
                for action_str in actions:
                    action = self.parse_action(action_str.strip())
                    self.player.dictStep(action)
                    print(f"Executed: {action_str}")
        except DoneException:
            print("Execution completed successfully!")
        finally:
            self.player.stop()

if __name__ == "__main__":
    # 示例数据加载执行
    # with open("data.json") as f:
    #     data = json.load(f)

    data = {
        "instruction": "place a cup with a knife in it on the kitchen counter space",
        "action": {
                    "0": "TeleportObject-countertop",
                    "1": "PickupObject-ButterKnife",
                    "2": "TeleportObject-cup",
                    "3": "OpenObject-Cabinet,PutObject-ButterKnife",
                    "4": "PickupObject-Cup,CloseObject-Cabinet",
                    "5": "TeleportObject-countertop",
                    "6": "PutObject-Cup"
        },
        "scene": "FloorPlan10"
    }
    
    executor = ActionExecutor()
    executor.execute(
        scene=data["scene"],
        plan=data["action"]
    )

    