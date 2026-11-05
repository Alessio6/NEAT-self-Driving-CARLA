@echo off
rem Watch a trained car drive, using the "carla" conda env's Python.
rem Any options are passed on to play.py, e.g.:
rem   play.bat --npcs 0 --episodes 5
rem   play.bat --viz --speed 2
rem   play.bat --genome models\winner_genome.pkl
cd /d "%~dp0"
"%USERPROFILE%\anaconda3\envs\carla\python.exe" -u play.py %*
