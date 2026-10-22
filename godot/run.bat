@echo off
rem Build C# and launch the game (without the editor)
cd /d "%~dp0"
dotnet build -nologo -v q
if errorlevel 1 (
    pause
    exit /b 1
)
start "" "D:\tools\godot\Godot_v4.7.2-stable_mono_win64\Godot_v4.7.2-stable_mono_win64.exe" --path "%~dp0."
