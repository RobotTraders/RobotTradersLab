import pytz

USE_TIMEZONE_AWARE: bool = True
DEFAULT_TIMEZONE = pytz.UTC
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S%z" if USE_TIMEZONE_AWARE else "%Y-%m-%d %H:%M:%S"
