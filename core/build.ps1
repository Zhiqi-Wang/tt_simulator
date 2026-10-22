# 编译 C++ 物理核心 core/build/ppcore.dll（MinGW g++，静态链接，只依赖系统 DLL），并复制给 Godot 工程
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PATH = "D:\msys64\mingw64\bin;$env:PATH"
New-Item -ItemType Directory -Force "$here\build" | Out-Null
# -ffp-contract=off：不做乘加融合，结果与 numba 版逐位一致（训练和游戏才能共用模型）
g++ -O3 -std=c++17 -ffp-contract=off -fno-fast-math -fopenmp -shared -static -static-libgcc -static-libstdc++ `
    -o "$here\build\ppcore.dll" "$here\ppcore.cpp"
$dst = Join-Path (Split-Path -Parent $here) "godot\bin"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$here\build\ppcore.dll" $dst -Force
Write-Host "built $here\build\ppcore.dll -> $dst"
