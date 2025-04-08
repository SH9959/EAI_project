def GotoObject(self, target_place: str):
    found_objects, feed_back_msg = self._findObject(target_place)
    
    if not found_objects:  # 检查是否找到对象
        raise Exception(f"Object '{target_place}' not found in the scene.")  # 或者处理错误信息

    obj = found_objects[0]  # 取第一个找到的对象
    position, rotation, horizon = self._getTeleportPose(obj)
    
    template = {
        "action": "Teleport",
        "position": position,
        "rotation": rotation,
        "horizon": horizon,
        "standing": True
    }
    return template, feed_back_msg 