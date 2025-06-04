class AppError(Exception):
    """应用基础异常"""
    pass

class CameraError(AppError):
    """摄像头相关错误"""
    pass

class ModelError(AppError):
    """模型相关错误"""
    pass

class ConfigError(AppError):
    """配置相关错误"""
    pass 