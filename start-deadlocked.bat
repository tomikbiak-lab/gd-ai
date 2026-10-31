@echo off
rem Select Deadlocked explicitly; do not review other levels.
call "%~dp0start-brain.bat" --level 20 %*
exit /b %errorlevel%
