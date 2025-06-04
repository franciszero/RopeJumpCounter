class JumpCounter:
    """跳绳计数核心逻辑"""
    
    def __init__(self):
        self.jump_cnt = 0
        self.jump_cnt_binary_mark = 0  # start with 000
        
    def process_prediction(self, prob: float, threshold: float) -> tuple[bool, int]:
        """
        处理预测结果，返回(是否正在上升, 当前跳数)
        """
        y_pred = int((prob > threshold))
        mark1 = (self.jump_cnt_binary_mark << 1) & 0b1111  # 保留最后3位
        self.jump_cnt_binary_mark = (mark1 | y_pred) & 0b1111
        
        if self.jump_cnt_binary_mark in [7, 15]: # [3, 7]:  # 3:011 -> 7:111，0111->1111
            is_on_rising = True
            if self.jump_cnt_binary_mark == 7:  # 3:  # 只有事件 3 检测为起跳事件
                self.jump_cnt += 1
        else:  # 0:000, 1:001, 2:010, 4:100, 5:101, 6:110
            is_on_rising = False
            
        return is_on_rising, self.jump_cnt
        
    def get_count(self) -> int:
        """获取当前跳数"""
        return self.jump_cnt
        
    def reset(self):
        """重置计数器"""
        self.jump_cnt = 0
        self.jump_cnt_binary_mark = 0 