@echo off
cd /d D:\learnPython\clean-ui
call venv\Scripts\activate  :: Activates the virtual environment
python clean-ui.py  :: Runs the Python script
pause  :: Keeps the window open after execution
