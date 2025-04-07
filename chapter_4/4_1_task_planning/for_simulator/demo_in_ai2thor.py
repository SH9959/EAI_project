import cv2
import json
from http import HTTPStatus
import dashscope

from myController import MyController, get_all_objects_in_scene, get_prompt, parse_action

# API key for the GLM large model
# dashscope.api_key = 'sk-37e58610f5544d878ac51c0920ca52b5'  # Replace with your API key 

# User instruction
instruction = "place a cup with a knife in it on the kitchen counter space"
scene = 'FloorPlan10'

# Initialize the AI2THOR controller
controller = MyController()
# Reset the scene
event = controller.reset(scene)
# Get all objects in the scene
all_objects = get_all_objects_in_scene(event)
print(all_objects)

# Get the skill set and their simple descriptions
with open("action.json", "r", encoding="utf-8") as f:
    skill_set = json.load(f)
# Initialize the history list
history = []

prompt = get_prompt(instruction, all_objects, skill_set)

SKILLS = [item['name'].lower() for item in skill_set]

# 在文件顶部添加IllegalException的定义
class IllegalException(Exception):
    """自定义异常类，用于处理非法操作"""
    pass

def task_is_not_finished(
    instruction:str,
    event
):
    """Determine whether the current task is completed based on pre-set task completion conditions, here only a simple condition is set
    """
    if instruction == "place a cup with a knife in it on the kitchen counter space":
    # Check if the last action is PutObject and if the object was successfully placed
        print(event)
        if event and event.metadata['lastAction'] == 'PutObject' and event.metadata['actionReturn']['placed']:
            return False
        return True

image = controller.getImg()

img_path = "tmp_image.png"  


# 保存图像到指定路径
cv2.imwrite(img_path, image)


# Multi-turn dialogue process
next_action_object = ""
while not 'done' in next_action_object.lower(): # task_is_not_finished(instruction,event):
    # Get the prompt
    prompt = get_prompt(instruction, all_objects, skill_set, history_of_actions=history, current_image=img_path)
    # print(prompt)

    # input("Press Enter to continue")



    messages = [
    {
        "role": "system",
        "content": [
        {"text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
        {"image": img_path},
        {"text": prompt}]
    }]

    response = dashscope.MultiModalConversation.call(
        # If no environment variable is configured, please replace the following line with: api_key ="sk-xxx"
        api_key = "sk-7ebcb16ed33147b69272c31c5568b1b5",# os.getenv('DASHSCOPE_API_KEY'),
        model = 'qwen-vl-plus',
        messages = messages
    )
    # Check the response status
    if response.status_code == HTTPStatus.OK:
        # Extract the next action
        next_action_object = response.output.choices[0].message.content[0]["text"]

        print(f"Next action: {next_action_object}")

        # parse_action
        next_action,object_id = parse_action(next_action_object)
        if not next_action.lower() in SKILLS:
            raise IllegalException(f"Illegal action: {next_action_object}")

        else:
            print(f"Action '{next_action_object}' is legal.")
            img, event = controller.execute(next_action_object)
            if event.metadata["lastActionSuccess"] == True:
                history.append(next_action_object)

            cv2.imwrite(img_path, img)  # 使用OpenCV保存图像
            print(f"Image saved to {img_path}")
    else:
        print('Failed to get a valid response from the model.')  
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response content: {response.content}")

    
# Print the history
print("History:", history)
