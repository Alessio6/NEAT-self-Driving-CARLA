@echo off
rem Runs play.py with the "carla" conda env's Python; any arguments are passed through.
rem   play.bat
rem   play.bat --npcs 0 --episodes 5
rem   play.bat --genome best_genome_no_traffic.pkl --npcs 0
cd /d "%~dp0"
"%USERPROFILE%\anaconda3\envs\carla\python.exe" -u play.py %*
