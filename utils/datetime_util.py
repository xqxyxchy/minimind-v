from datetime import datetime

def format_timedelta(start: datetime, end: datetime) -> str:
    delta = end - start
    total_seconds = delta.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours):02d}小时{int(minutes):02d}分{int(seconds):02d}秒"
    elif minutes > 0:
        return f"{int(minutes):02d}分{int(seconds):02d}秒"
    else:
        return f"{int(seconds):02d}秒"