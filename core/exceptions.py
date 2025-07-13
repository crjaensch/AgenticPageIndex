from typing import List

class PageIndexError(Exception):
    """Base exception for PageIndex operations"""
    pass

class PageIndexToolError(PageIndexError):
    """Exception raised by PageIndex tools"""
    def __init__(self, message: str, recovery_suggestions: List[str] = None):
        super().__init__(message)
        self.recovery_suggestions = recovery_suggestions or []

class PageIndexConfigError(PageIndexError):
    """Exception raised for configuration issues"""
    pass

class PageIndexFileError(PageIndexError):
    """Exception raised for file operations"""
    pass
