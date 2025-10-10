@echo off
echo Installing Chess Bot Dependencies...
echo ====================================

echo Installing pygame...
pip install pygame>=2.1.0

echo Installing python-chess...
pip install python-chess>=1.999

echo Installing numpy...
pip install numpy>=1.21.0

echo.
echo Installation complete!
echo.
echo To run the chess bot, execute:
echo python chess.py
echo.
echo Controls:
echo - Click to select and move pieces
echo - Click on difficulty line to change AI strength
echo - Press 'R' to reset game
echo - Press 'F' to flip colors
echo - Game analysis will be shown after each game
echo.
pause
