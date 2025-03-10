import ai2thor.controller
import dashscope
from http import HTTPStatus
# GLM大模型的API密钥
dashscope.api_key = 'sk-7ebcb16ed33147b69272c31c5568b1b5'  # 替换为你的API密钥 sk-7ebcb16ed33147b69272c31c5568b1b5

# 初始化AI2THOR控制器
controller = ai2thor.controller.Controller(scene='FloorPlan28')

# 重置场景
event = controller.reset(scene='FloorPlan28')
l = len(event.metadata['objects'])
all_objects = [event.metadata['objects'][i]['name'] for i in range(l)]
print(all_objects)

# 定义技能集合和其简单描述
skills = {
    'MoveAhead': "向前走一段指定的距离",
    'MoveBack': "向后走一段指定的距离",
    'MoveRight': "向右走一段指定的距离",
    'MoveLeft': "向左走一段指定的距离",
    'LookUp': "向上看指定度数",
    'LookDown': "向下看指定度数",
    'RotateRight': "向右旋转指定度数",
    'RotateLeft': "向左旋转指定度数",
    'PickupObject': "捡起一个物体,需要传入一个参数，比如apple",
    'PutObject': "将手中物体放到另一个物体上或指定的位置，需要传入一个参数，表示要放入的容器，比如水槽"
}

SKILLS = [item.upper() for item in skills.keys()]

# 初始化历史记录列表
history = []

# 用户指令
instruction = "请把毛巾放到桌子上"

model_name = 'qwen-turbo'
print(f"准备使用{model_name}")
def task_is_not_finished(
    instruction:str,
    event
):
    """根据预先设定的任务完成条件来判断当前任务是否完成
    """
    if instruction == "请把毛巾放到桌子上":
    # 检查最后一个动作是否是PutObject，并且是否成功放置了物体
        print(event)
        if event and event.metadata['lastAction'] == 'PutObject' and event.metadata['actionReturn']['placed']:
            return False
        return True

def get_id_by_object_str(objtype:str):

    found = []
    # agent_pos = self.last_event.metadata["position"]
    # agent_pos = np.array(agent_pos["x"],agent_pos["y"],agent_pos["z"])

    for obj in controller.last_event.metadata["objects"]:
        if(obj["objectType"].upper() == objtype.upper()):
            found.append(obj)

    return found[0]['objectId']

# 多轮对话过程
while 1: # task_is_not_finished():
    # 设计提示词
    prompt = f"""
你是一个智能家庭机器人，你的任务是根据自己的技能来完成用户的指令：“{instruction}”，
你已经完成的动作是：{history}，
需要根据自己的技能集合{skills}来完成用户下达的任务，
请输出下一步你应该采取的技能，你只需要输出英文的skill及其参数，比如PickupObject apple，而不要有任何额外的输出。
    """
    print(prompt)
    input("回车继续") # debug阶段


    # 为此你需要将用户的指令分解为若干技能步骤，每一个步骤都必须是你的技能集合中的一个，你的技能集合是：{skills}，

    # 调用GLM大模型
    messages = [{'role': 'system', 'content': f'You are a helpful assitant'},
                {'role': 'user', 'content': prompt}]

    response = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )

    # 检查响应状态
    if response.status_code == HTTPStatus.OK:
        # 提取下一步动作
        next_action = response.output.choices[0].message.content
        print(f"Next action: {next_action}")

        # 
        if next_action.upper() in SKILLS:

        # next_action有可能包含动作和物品，需要用空格split开，分别得到
        # 将next_action按空格分割成动作和物品
            action_parts = next_action.split(" ")
            if len(action_parts) > 1:
                next_action = action_parts[0]  # 动作名称
                object_id = action_parts[1]    # 物品名称
                # 更新动作参数
                event = controller.step(action=next_action, objectId=get_id_by_object_str(object_id))
                continue
            else:
                event = controller.step(action=next_action)

            if event.metadata["lastActionSuccess"] == True:
                history.append(next_action)
            else:
                print(f"执行动作失败: {next_action}")
        else:
            print(f"Action '{next_action}' is not in the skill set.")
    else:
        print('Failed to get a valid response from the model.')  

    

# 打印历史记录
print("历史记录:", history)
