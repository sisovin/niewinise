class Logger:
    LEVEL_INFO = 'INFO'
    LEVEL_ERROR = 'ERROR'

    def __init__(self, log_file, level):
        self.log_file = log_file
        self.level = level
        # Ensure the log file is created
        with open(self.log_file, 'a', encoding='utf-8') as f:
            pass

    def _write_log(self, level, message):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f'{level}: {message}\n')

    def info(self, message):
        if self.level == self.LEVEL_INFO:
            self._write_log(self.LEVEL_INFO, message)

    def error(self, message):
        self._write_log(self.LEVEL_ERROR, message)

    def set_level(self, level):
        self.level = level