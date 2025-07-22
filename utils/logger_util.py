"""
logger_util.py
日志格式化与异常日志工具模块。
提供彩色控制台日志、文件日志、自动分割与保留、结构化异常日志输出。
"""
import logging  # 导入日志模块，用于记录日志
import os  # 导入操作系统模块，用于文件和目录操作
from datetime import datetime  # 导入日期时间模块，用于生成时间戳
import glob  # 导入文件匹配模块，用于查找日志文件
import traceback  # 导入异常跟踪模块，用于获取异常堆栈信息
import sys  # 导入系统模块，用于获取异常信息

# 定义日志颜色和样式
class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器，用于在控制台输出带颜色的日志信息"""
    grey="\x1b[38;21m"  # 灰色
    blue="\x1b[38;5;39m"  # 蓝色
    yellow="\x1b[38;5;226m"  # 黄色
    red="\x1b[38;5;196m"  # 红色
    bold_red="\x1b[31;1m"  # 粗体红色
    green="\x1b[32m"  # 绿色
    purple="\x1b[35m"  # 紫色
    cyan="\x1b[36m"  # 青色
    reset="\x1b[0m"  # 重置颜色
    bold="\x1b[1m"  # 粗体
    underline="\x1b[4m"  # 下划线
    blink="\x1b[5m"  # 闪烁
    def __init__(self, fmt):
        super().__init__()  # 调用父类初始化方法
        self.fmt=fmt  # 保存格式化字符串
        self.FORMATS= {
            logging.DEBUG: self.grey+self.fmt+self.reset,  # 调试日志使用灰色
            logging.INFO: self.blue+self.bold+self.fmt+self.reset,  # 信息日志使用蓝色
            logging.WARNING: self.yellow+self.bold+self.fmt+self.reset,  # 警告日志使用黄色
            logging.ERROR: self.red+self.bold+self.fmt+self.reset,  # 错误日志使用红色
            logging.CRITICAL: self.bold_red+self.blink+self.fmt+self.reset  # 严重错误日志使用粗体红色
        }
    def format(self, record):
        # 对错误日志进行特殊格式化
        if record.levelno>=logging.ERROR and hasattr(record, 'structured_error'):
            return record.structured_error  # 返回结构化错误信息
        log_fmt=self.FORMATS.get(record.levelno)  # 获取对应日志级别的格式化字符串
        formatter=logging.Formatter(log_fmt)  # 创建格式化器
        return formatter.format(record)  # 返回格式化后的日志信息

def get_logger(name: str=None, level: int=logging.DEBUG, file_size: int=5, file_count: int=7, log_dir: str="logs", log_file: str=None):
    """获取带彩色控制台和文件输出的logger，自动分割和保留日志文件。"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 如果目录不存在，则创建

    if log_file is None:
        log_file=os.path.join(log_dir, 'minimind-v.log')  # 日志文件路径
    else:
        log_file=os.path.join(log_dir, log_file)  # 日志文件路径

    logger=logging.getLogger(name)  # 获取logger实例
    logger.setLevel(level)  # 设置日志级别为DEBUG
    
    if not logger.handlers:
        console_handler=logging.StreamHandler()  # 创建控制台处理器
        console_handler.setLevel(level)  # 设置控制台日志级别
        console_formatter=ColoredFormatter('%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s')  # 设置控制台日志格式
        console_handler.setFormatter(console_formatter)  # 应用格式化器
        logger.addHandler(console_handler)  # 添加控制台处理器
        file_handler=logging.FileHandler(log_file, encoding='utf-8')  # 创建文件处理器
        file_handler.setLevel(level)  # 设置文件日志级别
        file_formatter=logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s'
        )  # 设置文件日志格式
        file_handler.setFormatter(file_formatter)  # 应用格式化器
        logger.addHandler(file_handler)  # 添加文件处理器
        # 检查日志文件大小，超过xMB自动转存
        if os.path.getsize(log_file) > file_size * 1024 * 1024:  # MB
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')  # 生成时间戳
            new_log_file=os.path.join(log_dir, f'minimind-v_{timestamp}.log')  # 新日志文件路径
            os.rename(log_file, new_log_file)  # 重命名日志文件
        # 保留最多file_count份日志文件
        log_files=glob.glob(os.path.join(log_dir, 'minimind-v_*.log'))  # 获取所有日志文件
        if len(log_files) >file_count:
            log_files.sort(key=os.path.getctime)  # 按创建时间排序
            for old_log in log_files[:-file_count]:
                os.remove(old_log)  # 删除多余的日志文件
    return logger  # 返回logger实例

def log_exception(logger, exc_info=None, context_msg=None):
    """结构化输出异常日志，包含详细堆栈和上下文信息。"""
    if exc_info is None:
        exc_info=sys.exc_info()  # 获取当前异常信息
    tb_lines=traceback.format_exception(*exc_info)  # 格式化异常堆栈信息
    last_tb=exc_info[2]  # 获取最后一个堆栈帧
    while last_tb and last_tb.tb_next:
        last_tb=last_tb.tb_next  # 遍历堆栈帧
    if last_tb:
        filename=last_tb.tb_frame.f_code.co_filename  # 获取文件名
        funcname=last_tb.tb_frame.f_code.co_name  # 获取函数名
        lineno=last_tb.tb_lineno  # 获取行号
    else:
        filename=funcname=lineno='未知'  # 如果无法获取，则标记为未知
    logger.error(f"[{exc_info[0].__name__}] {exc_info[1]}")  # 输出标准错误日志
    structured_error= (
        f"\n\x1b[31;1m{'='*40} 错误发生 {'='*40}\n"
        f"异常类型: {exc_info[0].__name__}\n"
        f"异常信息: {exc_info[1]}\n"
        f"出错位置: {filename} -> {funcname}() -> 第{lineno}行\n"
        f"堆栈信息:\n{''.join(tb_lines)}"
        f"{'='*90}\x1b[0m\n"
    )  # 构建结构化错误信息
    logger.error('', extra={'structured_error': structured_error})  # 输出结构化错误日志
    
if __name__=="__main__":
    logger=get_logger(__name__)
    
    # 基础日志类型
    logger.debug("这是一条调试日志")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.critical("这是一条严重错误日志")
    
    # 带变量的日志
    user_id="hbgslz"
    action="登录"
    logger.info(f"用户 {user_id} 执行了 {action} 操作")
    
    # 异常日志
    try:
        1/0
    except Exception as e:
        log_exception(logger, context_msg="测试除零异常")
    
    # 性能日志
    import time
    start_time=time.time()
    time.sleep(1)
    logger.info(f"操作耗时: {time.time() -start_time:.2f}秒")
    
    # 业务日志
    logger.info("开始处理视频数据")
    logger.info("视频数据解析完成")
    logger.warning("视频数据不完整，部分字段缺失")
    
    # 系统状态日志
    try:
        import psutil
        logger.info(f"CPU使用率: {psutil.cpu_percent()}%")
        logger.info(f"内存使用率: {psutil.virtual_memory().percent}%")
    except ImportError:
        logger.warning("psutil模块未安装，无法获取系统状态")
    
    # 网络请求日志
    logger.info("开始请求抖音API")
    logger.info("API请求成功")
    logger.error("API请求失败: 网络超时")
    
    # 数据验证日志
    logger.info("开始验证数据格式")
    logger.warning("数据格式不符合预期")
    logger.error("数据验证失败: 必填字段缺失")
    
    # 配置相关日志
    logger.info("加载配置文件成功")
    logger.warning("配置文件部分参数缺失，使用默认值")
    logger.error("配置文件格式错误")
    
    # 安全相关日志
    logger.warning("检测到异常登录尝试")
    logger.error("安全验证失败")
    logger.critical("检测到潜在的安全威胁")