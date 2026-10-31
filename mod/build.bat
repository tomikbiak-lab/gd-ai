@echo off
setlocal
rem Run from an x64 Visual Studio Developer Command Prompt.
rem Set GEODE_SDK to your Geode 5.11.2 SDK directory first.
if not defined GEODE_SDK (
    echo ERROR: Set GEODE_SDK to your Geode SDK directory.
    exit /b 1
)
where cl >nul 2>&1 || (
    echo ERROR: Run this from an x64 Visual Studio Developer Command Prompt.
    exit /b 1
)
where cmake >nul 2>&1 || exit /b 1
where ninja >nul 2>&1 || exit /b 1
where geode >nul 2>&1 || exit /b 1
cd /d "%~dp0"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSKIP_BUILDING_CODEGEN=OFF || exit /b 1
cmake --build build --target GDAI_PACKAGE --parallel 2 || exit /b 1
echo Built build\takto.gd-ai.geode. Copy it into your game's geode\mods folder to install.
